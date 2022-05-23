# -*- coding: utf-8 -*-
# @Time:2021/10/26 10:55:24
# @File:pspnet.py
# @Author:Jianyuan Hong
# @Mail: hongjianyuan1997@gmail.com

"""
    Reference:
       
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50,resnet101,resnet152
from .backbone.resnet_v1b import resnet50_v1s,resnet101_v1s,resnet152_v1s
from torchsummaryX import summary
from thop import profile

__all__ = ['PspNet', 'get_pspnet',]

class PspNet(nn.Module):
    r"""

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    """
    def __init__(self, nclass,**kwargs):
        super(PspNet, self).__init__()
        # 内部组件
        self.head = _PSPHead(nclass, **kwargs)
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



def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )

class _PSPHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)

class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

def get_pspnet(dataset='citys'):
    from ..data.dataloader import datasets
    model = PspNet(datasets[dataset.lower()].NUM_CLASS)
    return model


if __name__ == '__main__':
    img_1 = torch.randn(2, 3, 512, 512)
    model = PspNet(6)
    macs, params = profile(model, inputs=(img_1, ))
    print(macs, params)
    print("=================================================")
    summary(model, img_1)
    outputs = model(img_1)
