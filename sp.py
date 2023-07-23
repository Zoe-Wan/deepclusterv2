import torch
from network.head_p import LinearHead_p
from util.torch_dist_sum import *
from util.meter import *
import time
from network.backbone import *
from util.accuracy import accuracy
from dataset.data import *
import torch.nn.functional as F
from util.dist_init import dist_init
from torch.nn.parallel import DistributedDataParallel
import argparse
import math
from util.DistributedSamplerWrapper import DistributedSamplerWrapper
from util.UnifLabelSampler import UnifLabelSampler
from dataset.imagenet import Imagenet
from util.mixup import Mixup
from torchvision import datasets
import spclustering
import os
from util.Logger import Logger
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23457)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--k',type=int,default=100)
parser.add_argument('--ngpu',type=int,default=4)
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--use_fp16', default=False, action='store_true')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--reassign',type=int,default=1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--T', type=float,default=0.07)
args = parser.parse_args()

epochs = args.epochs
warm_up = 10

mix_func = Mixup(mixup_alpha=1, cutmix_alpha=1, switch_prob=1, label_smoothing=0)

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr  * T / warmup_iters
    else:
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = 0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * base_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler, alpha, T):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses1, losses2, losses],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    # switch to train mode
    model.train()

    for i, (samples, targets, ptargets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, lr, i, len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)
        
        samples = samples.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)
        ptargets = targets.cuda(local_rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            output1, output2 = model(samples, True)
            loss1 = F.cross_entropy(output1, targets)
            loss2 = F.cross_entropy(output2/T, ptargets)
            loss = loss2*alpha+loss1*(1-alpha)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses1.update(loss1.item(), samples.size(0))
        losses2.update(loss2.item(), samples.size(0))
        losses.update(loss.item(), samples.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)



@torch.no_grad()
def test(test_loader, model, local_rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output, _ = model(img, True)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc


def main():
    rank, local_rank, world_size = dist_init(args.port)
    batch_size = args.bs // world_size
    num_workers = 6
    lr = args.lr * batch_size * world_size / 256
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0) 
    if rank == 0:
        print(args)
    
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100


    model = backbone_dict[args.backbone]()
    model = LinearHead_p(net=model, dim_in=dim_dict[args.backbone], dim_out=num_classes,pseudo = args.k, fix=False)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wd, momentum=0.9)

    torch.backends.cudnn.benchmark = True
    train_aug, test_aug = get_train_augment(args.dataset), get_test_augment(args.dataset)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='data', download=True, transform=train_aug)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_aug)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='data', download=True, transform=train_aug)
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=test_aug)

    
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #     num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    if rank == 0:
        cluster_log = Logger(os.path.join('checkpoints/','clusters'))
    if not os.path.exists('checkpoints') and rank == 0:
        os.makedirs('checkpoints')

    checkpoint_path = os.path.join('checkpoints/', args.checkpoint)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    best_top1 = 0
    best_top5 = 0
    
    dc = spclustering.Kmeans(args.k)
    

    for epoch in range(start_epoch, epochs):
        # train_sampler.set_epoch(epoch)
        features = dc.compute_features(train_loader, model, len(train_dataset), batch_size, rank,local_rank)
        # features_list = [features for _ in range(world_size)]
        print(features[0][0])
        # torch.distributed.all_gather(features_list, features, group=None, async_op=False)
         
        dc.cluster(features,args.ngpu,rank,epoch)
	


        assigned_dataset = dc.cluster_assign(train_dataset) 
        sampler = UnifLabelSampler(len(assigned_dataset), dc.images_lists)
        assigned_sampler = DistributedSamplerWrapper(sampler)

        assigned_loader = torch.utils.data.DataLoader(
        assigned_dataset, batch_size=batch_size, shuffle=(assigned_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=assigned_sampler)

        # assigned_loader = torch.utils.data.DataLoader(
        #     assigned_dataset, batch_size=batch_size, num_workers=num_workers,
        #     sampler=sampler, pin_memory=True )

        
        # pseudo = len(dc.images_lists)
        train(assigned_loader, model, local_rank, rank, optimizer, lr, epoch, scaler, args.alpha, args.T)
        top1, top5 = test(test_loader, model, local_rank)
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)
        print(len(dc.images_lists[dc.arrange_clustering(dc.images_lists)[0]])) 
        if rank == 0:
            print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, top1, top5, best_top1, best_top5))
            
            try:
                nmi = normalized_mutual_info_score(
                    dc.arrange_clustering(dc.images_lists),
                    dc.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            state_dict =  {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch + 1
            }

            torch.save(state_dict, checkpoint_path)
            cluster_log.log(dc.images_lists)

if __name__ == "__main__":
    main()





