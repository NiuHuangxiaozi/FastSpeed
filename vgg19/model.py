#
# Implementation of AlexNet for illustrative purposes. The train.py driver
# can import AlexNet from here or directly from torchvision.
#
# Taken from torchvision.models.alexnet:
# https://pytorch.org/docs/1.6.0/_modules/torchvision/models/alexnet.html

import torch.nn as nn
from torchvision import models

class vgg19(nn.Module):
    def __init__(self,num_classes):
        super(vgg19, self).__init__()
        self.vgg19=models.vgg19(pretrained=False,num_classes=num_classes)
    def forward(self, x):
        x=self.vgg19(x)
        return x


