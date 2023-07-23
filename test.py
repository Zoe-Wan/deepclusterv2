import torch 
from util import AverageMeter, Logger, UnifLabelSampler

def test(loader, model, class_num):

    batch_time = AverageMeter()
    losses = AverageMeter()
    num_correct = 0
    # switch to eval mode
    model.eval()

    pre_num = torch.zeros((1, class_num))
    tar_num = torch.zeros((1, class_num))
    acc_num = torch.zeros((1, class_num))

    for i, (input_tensor, class_target) in enumerate(loader):


        class_target = class_target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())


        class_target_var = torch.autograd.Variable(class_target)
        # todo
        _,output = model(input_var, True)

        pred = output.argmax(dim=1)
        # batch_num_correct = torch.eq(pred, class_target).sum().float().item()
        # num_correct += batch_num_correct
        pre_mask = torch.zeros(output.size()).scatter_(1,pred.cpu().view(-1,1),1.)
        pre_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1,class_target.cpu().view(-1,1),1.)
        tar_num +=tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)

    recall = acc_num/tar_num
    precision = acc_num/pre_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/tar_num.sum(1)
    print('Test:' 
          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
          'Recall: {0}\t'
          'F1: {1}\t'
          'Accuracy: {2}'
          .format(recall.data.numpy()[0], F1.data.numpy()[0], accuracy.data.numpy()[0], loss=losses))
