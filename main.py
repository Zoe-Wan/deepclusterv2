# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.resnet18 import resnet18 
import models
import clustering
from util import AverageMeter, Logger, UnifLabelSampler
from train import train
from test import test



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--port',type=int)
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    # 多少个epoch重新聚类一次
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')                  
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--alpha', type=float, default=0, help='pseudo label loss weight (default 0)')
    parser.add_argument('--ngpu', type=int, default=1, help='gpu number (default 1)')
    # pseudo loss大约是class loss的十倍左右
    # ...verbose的意思是开启后台日志输出
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()


def main(args):
    batch_size = args.batch
    num_workers = args.workers
    rank, local_rank, world_size = dist_init(args.port)
    devices_ids = [i for i in range(args.ngpu)]
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # preprocessing of data


    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    # load the data
    end = time.time()

    train_dataset = datasets.CIFAR10(root=args.data,train=True,transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data,train=False,transform=transform_test)
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    # 选择模型框架
    # model = models.__dict__[args.arch](sobel=args.sobel, pseudo=args.nmb_cluster, out=len(train_dataset.classes))
    resnet = resnet18(len(train_dataset.classes))
    model = models.Net(resnet, pseudo_num_classes=args.nmb_cluster, num_classes=len(train_dataset.classes))
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # optimizer_pseudo_tl.load_state_dict(checkpoint['optimizer_pseudo'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # cluster_log = Logger(os.path.join('test/','clusters'))
    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    dc = clustering.Kmeans(args.nmb_cluster)



    for epoch in range(args.start_epoch, args.epochs):

        features = dc.compute_features(train_loader, model, len(train_dataset), args.batch)

        dc.cluster(features, args.ngpu,epoch)
        
        assigned_dataset = dc.cluster_assign(train_dataset) 
        sampler = UnifLabelSampler(len(assigned_dataset), dc.images_lists)
        assigned_sampler = DistributedSamplerWrapper(sampler)

        assigned_loader = torch.utils.data.DataLoader(
        assigned_dataset, batch_size=batch_size, shuffle=(assigned_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=assigned_sampler)

        # loss = train(train_dataloader, model, criterion, 
        # optimizer, optimizer_pseudo_tl, optimizer_tl,
        # epoch, args, deepcluster, len(train_dataset.classes))
        
        train(assigned_loader, model, criterion, optimizer, epoch, args,  len(train_dataset.classes))
        test(test_loader, model,len(test_dataset.classes))
        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'.format(epoch))
            print('####################### \n')
        try:
            nmi = normalized_mutual_info_score(
                dc.arrange_clustering(dc.images_lists),
                dc.arrange_clustering(cluster_log.data[-1])
            )
            print('NMI against previous assignment: {0:.3f}'.format(nmi))
        except IndexError:
            pass
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),

                   },os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(dc.images_lists)
    



if __name__ == '__main__':
    args = parse_args()
    main(args)
