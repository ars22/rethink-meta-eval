import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.dropblock import DropBlock
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from collections import defaultdict



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, activation='ReLU'):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = z_dim * 25


class ShallowConv(nn.Module):
    def __init__(self, 
        h_dim,
        z_dim, 
        projection, 
        classifier_type,
        num_classes,
        x_dim=3,
        retain_last_activation=True,
        activation='ReLU',
        learnable_scale=False):
        
        super(ShallowConv, self).__init__()
        
        self.encoder = nn.Sequential(
            conv_block(x_dim, h_dim),
            conv_block(h_dim, h_dim),
            conv_block(h_dim, h_dim),
            conv_block(h_dim, z_dim),
        )
        
        # classifier creation
        self.projection = projection
        print("Unit norm projection is ", self.projection)
        print("Avg pool is always False for Conv64")
        self.final_feat_dim = z_dim * 25
        self.num_classes = num_classes
        self.no_fc_layer = (classifier_type == "no-classifier")
        self.classifier_type = classifier_type

        if self.no_fc_layer is True:
            self.fc = None
        elif self.classifier_type == 'linear':
            self.fc = nn.Linear(self.final_feat_dim, num_classes)
            self.fc.bias.data.fill_(0)
        else:
            raise ValueError("classifier type not found")

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # init learnable scale
        if learnable_scale:
            self.scale = torch.nn.Parameter(torch.FloatTensor([1.0]))



    def forward(self, x, only_features=False):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # hypersphere projection
        if self.projection:
            x_norm = torch.norm(x, dim=1, keepdim=True)+0.00001
            x = x.div(x_norm)

        # fc
        if (self.fc is not None) and (not only_features):
            x = self.fc.forward(x)
        
        return x

