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

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        drop_rate=0.0,
        drop_block=False,
        block_size=1):
        
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)



    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, 
            block, 
            classifier_type,
            avg_pool,
            drop_rate,
            dropblock_size,
            projection, 
            num_classes, 
            keep_prob=1.0,
            learnable_scale=False):

        self.inplanes = 3
        super(ResNet, self).__init__()
        self.projection = projection
        print("Unit norm projection is ", self.projection)
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(dropblock_size, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        print("Average pooling: ", self.keep_avg_pool)


        # classifier creation
        self.final_feat_dim = 640
        self.classifier_type  = classifier_type
        self.num_classes = num_classes
        self.no_fc_layer = (classifier_type == "no-classifier")
        
        if self.no_fc_layer is True:
            self.fc = None
        elif classifier_type == 'linear':
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


    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, only_features=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # hypersphere projection
        if self.projection:
            x_norm = torch.norm(x, dim=1, keepdim=True)+0.00001
            x = x.div(x_norm)

        # fc
        if (self.fc is not None) and (not only_features):
            x = self.fc.forward(x)
        
        return x


def resnet12(**kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(block=BasicBlock, **kwargs)
    return model



