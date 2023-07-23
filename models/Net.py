from torchvision.models.resnet import resnet18
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, model, pseudo_num_classes, num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        # print(self.features)
        # 目前只能接受model为resnet18，要怎么自适应输入尺寸明天问
        self.pseudo_top_layer = nn.Linear(512, pseudo_num_classes)
        self.top_layer = nn.Linear(512, num_classes)

       

    def forward(self, x, flag):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        y = self.top_layer(x)
        pseudo_y = self.pseudo_top_layer(x)
        if flag:
          return pseudo_y, y
        return x

