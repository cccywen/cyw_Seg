#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 9:51
# @Author:Jianyuan Hong
# @File:inference.py
# @Software:Vscode

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import numpy as np
from PIL import Image
from core.models.model_zoo import get_segmentation_model


parser = argparse.ArgumentParser(
    description='inference segmentation result from a given image or files')
parser.add_argument('--model', type=str, default='pspnet',
                    choices=['danet', 'pspnet'], help='model name (default: fcn32s)')
parser.add_argument('--dataset', type=str, default='potsdam',
                    choices=['vaihingen', 'potsdam'], help='dataset name (default: pascal_voc)')
parser.add_argument('--checkpoint', default="/home/caoyiwen/slns/XLPR_Semantic_Segmentation-master/runs/models/pspnet_potsdam_best_model.pth",
                    help='Directory for saving checkpoint models')
parser.add_argument('--input', type=str, default="/home/dataset/Remote_Sensing/Vaihingen/test.txt",
                    help='path to the input picture or path to the files or txt')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
parser.add_argument('--devices', type=str, default='0',
                    help='The ids of GPU to be used')
args = parser.parse_args()


vaihingenpalette = np.array([[0, 0, 0],#bg
                            [255, 255, 255],#is
                           [255, 255, 0],#car
                           [0, 0, 255],#build
                           [0, 255, 255],#low vege
                           [0, 255, 0]#tree
                           ], dtype='uint8').flatten()


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # 初始化模型
    model = get_segmentation_model(
        model=args.model, dataset=args.dataset).to(device)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    print('Finished loading model!')

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # # load data
    with open(args.input, 'r') as lines:
        for line in lines:
            _images = line.rstrip('\n')
            print(_images)
            assert os.path.isfile(_images)
            img = cv2.imread(_images)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            images = transform(img).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(images)
                pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                print(pred.shape)
                # 填色
                visualimg = Image.fromarray(pred.astype('uint8'),'P')
                visualimg.putpalette(vaihingenpalette) 
                visualimg.save('pspnet.png')
                break



    # files = os.listdir(args.input)
    # for file in files:
    #     image_path = os.path.join(args.input,file)
    #     img = cv2.imread(image_path)
    #     images = transform(img).unsqueeze(0).to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         output = model(images)
    #         pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    #         pred = np.where(pred == 1,255, 0)
    #         cv2.imwrite(args.outdir +'/'+os.path.basename(image_path), pred)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    inference(args)
