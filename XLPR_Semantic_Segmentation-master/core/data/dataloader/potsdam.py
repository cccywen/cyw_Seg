# -*- coding: utf-8 -*-
# @Time:2021/11/01 09:48:53
# @File:potsdam.py
# @Author:Jianyuan Hong
# @Mail: hongjianyuan1997@gmail.com

import os
import collections
import numpy as np
from PIL import Image
import cv2
from segbase import SegmentationDataset
import torch.utils.data as data
import torch

import shutil
import sys

class PotsdamSegmentation(SegmentationDataset):
    """
    Parameters
    ----------
    root : string
        Path to Foot Segmentation  folder. Default is './datasets/Foot Ulcer Segmentation Challenge'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'Potsdam'
    NUM_CLASS = 6  # Number of categories

    def __init__(self, root='/home/caoyiwen/data/', split='train', **kwargs):
        super(PotsdamSegmentation, self).__init__(root, split, **kwargs)
        self.root = os.path.join(root, self.BASE_DIR)

        o_path = os.getcwd()
        print(o_path)
        par_path = os.getcwd().split("slns")[0]+"data"
        print("par_path"+par_path)

        # print(self.root)
        if split == 'train':
            _split_f = os.path.join(self.root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(self.root, 'validation.txt')
        elif split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []

        with open(_split_f, 'r') as lines:
            for line in lines:

                _images = par_path+line.rstrip('\n')
                # print(_images)
                # print(os.path.isfile(_images))
                assert os.path.isfile(_images)
                self.images.append(_images)
                _mask = _images.replace('images', 'labels')
                _mask = _mask.replace('.jpg', '.png')
                # print(_mask)
                # print(os.path.isfile(_mask))
                assert os.path.isfile(_mask)
                # print("mask" + _mask)
                self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), self.root))

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        # mask_2 = Image.open(self.masks[index])

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # img_tr = Image.fromarray(mask)
        # 转变像素值
        mask = np.where(mask == 255,1,mask)
        mask = np.where(mask == 76,0,mask)
        mask = np.where(mask == 0,0,mask)
        mask = np.where(mask == 226,2,mask)
        mask = np.where(mask == 29,3,mask)
        mask = np.where(mask == 179,4,mask)
        mask = np.where(mask == 150,5,mask)
        mask = np.where(mask == 225,2,mask)
        # img_tr_2 = Image.fromarray(mask)
        # for j in img_tr.getcolors():
        #     if j[1] == 225:
        #         print('原始图像的像素：',mask_2.getcolors())
        #         print('原始灰度：',img_tr.getcolors())
        #         print('转换后：',img_tr_2.getcolors())
        #         print(i[2])
        #         print('---------------------------')

        if self.split == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'val':
            img, mask = self._sync_transform(img, mask)
        elif self.split == 'test':
            img, mask = self._sync_transform(img, mask)
        return img, mask, self.images[index]

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names"""
        return ('background', 'impervious surface', 'car', 'building', 'low vegetation', 'trees')


if __name__ == '__main__':
    dataset = PotsdamSegmentation(split='train')
    train_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True)
    for i in train_loader:
        aa = np.squeeze(i[1].numpy())
        img_tr = Image.fromarray(np.uint8(aa))
        for j in img_tr.getcolors():
            if j[1] > 5:
                print(i[2])
                print(img_tr.getcolors())
