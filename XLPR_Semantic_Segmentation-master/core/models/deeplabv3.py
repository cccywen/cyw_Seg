# -*- coding: utf-8 -*-
# @Time:2021/10/26 11:12:08
# @File:deeplabv3.py
# @Author:Jianyuan Hong
# @Mail: hongjianyuan1997@gmail.com

"""
    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
       
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50,resnet101,resnet152
from .backbone.resnet_v1b import resnet50_v1s,resnet101_v1s,resnet152_v1s
from torchsummaryX import summary
from thop import profile

__all__ = ['Deeplabv3', 'get_deeplabv3',]

class Deeplabv3(nn.Module):
    r"""

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    """
    def __init__(self, nclass,**kwargs):
        super(Deeplabv3, self).__init__()
        # 内部组件
        self.head =  _DeepLabHead(nclass, **kwargs)
        # 参数
        self.nclass = nclass # Number of categories
        # 可修改的
        self.backbone = 'resnet50_v1s'# backbone
        self.pretrained_base = True # bool if have pretrained weight

        if self.backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=self.pretrained_base, **kwargs)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=self.pretrained_base, **kwargs)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152(pretrained=self.pretrained_base, **kwargs)
        elif self.backbone == 'resnet50_v1s':
            self.pretrained = resnet50_v1s(pretrained=self.pretrained_base, **kwargs)
        elif self.backbone == 'resnet101_v1s':
            self.pretrained = resnet101_v1s(pretrained=self.pretrained_base, **kwargs)
        elif self.backbone == 'resnet152_v1s':
            self.pretrained = resnet152_v1s(pretrained=self.pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
    
    def forward(self, x):
        size = x.size()[2:]
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        outputs = []
        x = self.head(c4)
        x0 = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x0)
        return outputs

class _DeepLabHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

def get_deeplabv3(dataset='citys'):
    from ..data.dataloader import datasets
    model = Deeplabv3(datasets[dataset.lower()].NUM_CLASS)
    return model

if __name__ == '__main__':
    img_1 = torch.randn(2, 3, 512, 512)
    model = Deeplabv3(6)
    macs, params = profile(model, inputs=(img_1, ))
    print(macs, params)
    print("=================================================")
    summary(model, img_1)
    outputs = model(img_1)


