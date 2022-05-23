#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 10:14
# @Author:Jianyuan Hong
# @File:FUSeg.py
# @Software:PyCharm

import os
import torch
import scipy.io as sio
import numpy as np

from PIL import Image
import cv2
from .segbase import SegmentationDataset
import glob


class FootUlcerSegmentation(SegmentationDataset):
    """MICCAI 2021 Foot Ulcer Segmentation Challenge.
    Parameters
    ----------
    root : string
        Path to Foot Ulcer Segmentation Challenge folder. Default is './datasets/Foot Ulcer Segmentation Challenge'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = ''
    NUM_CLASS = 2  # Number of categories

    def __init__(self, root='F:\dataset', split='train'):
        super(FootUlcerSegmentation, self).__init__(root, split)
        self.root = os.path.join(root, self.BASE_DIR)

        if split == 'train':
            _split_f = os.path.join(self.root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(self.root, 'validation.txt')
        elif split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Uknown dataset split.')

        self.images = []
        self.masks = []

        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _images = line.rstrip('\n')
                assert os.path.isfile(_images)
                self.images.append(_images)
                _mask = _images.replace('images', 'labels')
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), self.root))

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.where(mask > 128, 1, 0)
        if self.split == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'val':
            img, mask = self._val_sync_transform(img, mask)

        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names"""
        return ('normal', 'injured')


if __name__ == '__main__':
    dataset = FootUlcerSegmentation(split='train')
    for image, mask, name in dataset:
        print(image.shape, mask.shape, name)
        print(mask)
