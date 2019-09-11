import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import SpatialAttentionModule, ChannelAttentionModule
from .base import BaseNet

# TODO: Use Dilated-Resnet50 from
# https://github.com/junfu1115/DANet/blob/master/encoding/dilated/resnet.py

# TODO: Test on CityScapes

class DANet(BaseNet):
    """
    Re-implementation of Dual Attention Network(Jun Fu etal)
    FCN style upsampling.
    """
    def __init__(self,
                 backbone,
                 nclass,
                 pretrain_root,
                 dilated,
                 aux_loss=False,
                 se_loss = False, # weighted loss with multi-scale output
                 norm_layer = nn.BatchNorm2d,
                 **kwargs):
        super(DANet, self).__init__(backbone,
                                    nclass,
                                    pretrain_root,
                                    dilated=dilated)
        self.head = DANetHead(2048, nclass, norm_layer)
    def forward(self, x):
        im_size = x.size()[2:]
        _, _, _, c_4 = self.extract_features(x)
        (sa, ca, saca) = self.head(c_4)
        sa = F.upsample(sa, im_size, **self.up_args)
        ca = F.upsample(ca, im_size, **self.up_args)
        saca = F.upsample(saca, im_size, **self.up_args)
        return (sa, ca, saca)





class DANetHead(nn.Module):
    """The actual part of the DANet
    Code from: https://github.com/junfu1115/DANet/blob/master/encoding/models/danet.py
    """
    def __init__(self, in_channel, out_channel, norm_layer):
        super(DANetHead, self).__init__()
        inter_channel = in_channel//4

        # spatial convs
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 3, padding=1, bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )

        self.conv5b = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 3, padding=1, bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )

        self.sa = SpatialAttentionModule(inter_channel)
        self.sc = ChannelAttentionModule()

        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )

        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )
        # prediction layers
        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channel, 1, padding=0)
        )
        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channel, 1, padding=0)
        )
        
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channel, 1, padding=0)
        )
    def forward(self, x):
        feat1 = self.conv5a(x)
        sf1 = self.sa(feat1)
        s_conv = self.conv51(sf1)
        s_out = self.conv6(s_conv)

        feat2 = self.conv5b(x)
        sf2 = self.sc(feat2)
        c_conv = self.conv52(sf2)
        c_out = self.conv7(c_conv)

        sasc_out = s_out + c_out  # sum fusion

        outputs = (s_out,
                   c_out,
                   sasc_out)
        return outputs

