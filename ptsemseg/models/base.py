import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.dilated_resnet import resnet50, resnet101

class BaseNet(nn.Module):
    """Base model that accepts
    all possible configs and runs the
    feature extraction.
    """
    def __init__(self,
                 backbone,
                 nclass,
                 pretrain_root,
                 dilated=True,
                 multi_dilation=False,
                 multi_grid=False):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.dilated = dilated
        self.multi_dilation = multi_dilation
        self.grid = multi_grid
        self.up_args = {
            'mode': 'bilinear',
            'align_corners': True
        }
        self.pretrained = None

        # TODO: choose the appropriate resnet model from pre-trained
        if backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=True,
                                       root=pretrain_root,
                                       dilated=dilated,
                                       multi_grid=multi_grid,
                                       multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=True,
                                        root=pretrain_root,
                                        dilated=dilated,
                                        multi_grid=multi_grid,
                                        multi_dilation=multi_dilation)
        else:
            raise RuntimeError('Unkown backbone:{}'.format(backbone))

    def extract_features(self, x):
        """Extract features from the base model"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c_1 = self.pretrained.layer1(x)
        c_2 = self.pretrained.layer2(c_1)
        c_3 = self.pretrained.layer3(c_2)
        c_4 = self.pretrained.layer4(c_3)

        return (c_1, c_2, c_3, c_4)


