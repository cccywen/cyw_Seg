#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 10:34
# @Author:Jianyuan Hong
# @File:segbase.py
# @Software:PyCharm

"""Base segmentation dataset with Data Augmentation"""
import cv2
import torch
from torchvision import transforms
# 数据增强包
from albumentations import (
    Compose, Resize, OneOf, RandomBrightness, RandomContrast, MotionBlur, MedianBlur,
    GaussianBlur, VerticalFlip, HorizontalFlip, ShiftScaleRotate, Normalize,
)

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.split = split
        self.transforms_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
        self.size = [512, 512]

    # 数据增强函数
    def albumentations_transform(self, img, masks):
        # 训练集的增强
        trans_train = Compose([
            # 随机更改输入图像的色相，饱和度和值
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            # 通过用255减去像素值来反转输入图像。
            # InvertImg(always_apply=False, p=1),
            # 随机改变RGB三个通道的顺序
            # ChannelShuffle(always_apply=False, p=0.5),
            # 随机出现小黑点
            # Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            # RandomCrop(224, 224),
            # OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            # OneOf([MotionBlur(blur_limit=3),MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3),], p=0.5,),
            VerticalFlip(p=0.5),
            # HorizontalFlip(p=0.5),
            # ShiftScaleRotate(
            #     shift_limit=0.2,
            #     scale_limit=0.2,
            #     rotate_limit=20,
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_REFLECT_101,
            #     p=1,
            # ),
        ])

        augmented = trans_train(image=img, mask=masks)
        # print(augmented)
        return augmented['image'], augmented['mask']

    def _val_sync_transform(self, img, mask):
        img, mask = self.albumentations_transform(img, mask)
        # final transform
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        img = self.transforms_data(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

    def _sync_transform(self, img, mask):
        img, mask = self.albumentations_transform(img, mask)
        # final transform
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        img = self.transforms_data(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
