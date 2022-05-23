#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 12:19
# @Author:Jianyuan Hong
# @File:unet.py
# @Software:PyCharm

"""U-Net"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet', 'get_unet']


# 两次卷积，如果有中间通道先卷积到中间通道再卷积到输出通道数
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样后续过程，最大池化图片大小减半，再用DoubleConv增加通道数
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # 上采样使用blinear或者转置卷积扩大特征图宽高为两倍。
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)  # x1上采样获得的特征,x2下采样获得的特征
        # input is CHW
        # 计算两张图长宽的差值，用于补充padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,  # F.pad（左填充，右填充，上填充，下填充）
                        diffY // 2, diffY - diffY // 2])
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # 拼接后卷积


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, nclass):  # nclass 对应类别数
        super(UNet, self).__init__()
        self.bilinear = True
        self.n_channels = 3

        self.inc = DoubleConv(self.n_channels, 64)  # 两次卷积
        self.down1 = Down(64, 128)  # 2 两次卷积后下采
        self.down2 = Down(128, 256)  # 4
        self.down3 = Down(256, 512)  # 8
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 16
        # 先把传入的特征图x上采样
        # 然后把x和之前的图concat，最后再进行两次卷积。
        # 这样即可在逐步把特征图大小变回原图的同时，逐步缩小通道数为64。
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)  # 逐步把特征图大小变回原图的同时，逐步缩小通道数为64
        self.outc = OutConv(64, nclass)  # 1*1卷积，通道由64转为类别数，每个类别对应一个mask

    def forward(self, x):
        # 两次卷积，第一次3到64维，第二次64到64
        x1 = self.inc(x)
        # 连续三次下采样，每次下采样都maxpool和两次卷积，特征图宽高为1/2
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outputs = list()
        logits = self.outc(x)  # 1*1卷积，通道由64转为类别数，每个类别对应一个mask
        outputs.append(logits)
        return tuple(outputs)


def get_unet(dataset='citys'):
    from ..data.dataloader import datasets
    model = UNet(datasets[dataset.lower()].NUM_CLASS)
    return model

