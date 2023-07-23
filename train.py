from util import AverageMeter, Logger, UnifLabelSampler
import os
import torch

def train(loader, model, crit, opt_model, epoch, args, class_num):
    print("train")


    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    # TP = 0
    # T = 0
    pre_num = torch.zeros((1, class_num))
    tar_num = torch.zeros((1, class_num))
    acc_num = torch.zeros((1, class_num))
    # switch to train mode
    model.train()



    for i, (input_tensor,class_target,p_target) in enumerate(loader):

        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt_model.state_dict(),
            }, path)



        class_target = class_target.cuda(non_blocking=True)
        p_target = p_target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())


        class_target_var = torch.autograd.Variable(class_target.cuda())
        p_target_var = torch.autograd.Variable(p_target.cuda())
        # todo
        
        p,output = model(input_var, True)


        loss1 = crit(output, class_target_var)
        loss2 = crit(p, p_target_var)

        loss = loss1*0.9+loss2*0.1
        losses.update(loss1.item(), input_tensor.size(0))

        pred = output.argmax(dim=1)

        # batch_TP = torch.eq(pred, class_target).sum().float().item()
        # TP += batch_TP

        pre_mask = torch.zeros(output.size()).scatter_(1,pred.cpu().view(-1,1),1.)
        pre_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1,class_target.cpu().view(-1,1),1.)
        tar_num +=tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)

        # compute gradient and do SGD step
        opt_model.zero_grad()

        loss.backward()
        opt_model.step()

    recall = acc_num/tar_num
    precision = acc_num/pre_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/tar_num.sum(1)
    print('Epoch: [{0}]\t'
          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
          'Recall: {1}\t'
          'F1: {2}\t'
          'Accuracy: {3}'
          .format(epoch, recall.data.numpy()[0], F1.data.numpy()[0], accuracy.data.numpy()[0], loss=losses))
    return losses.avg
