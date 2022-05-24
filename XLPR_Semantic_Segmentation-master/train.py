#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 9:51
# @Author:Jianyuan Hong
# @File:train.py

import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
from apex import amp
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.loss.loss import get_segmentation_loss
from core.utils.logger import setup_logger
from core.utils.optimzer import get_optimizer
from core.utils.lr_scheduler import get_lr_scheduler
from core.utils.score import SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')  # 初始化解析器
    # model and dataset
    parser.add_argument('--model', type=str, default='resuneta_py',  # 定义参数
                        choices=['unet', 'resuneta_py', 'pfsegnet', 'pspnet'], help='model name (default: fcn32s)')
    parser.add_argument('--dataset', type=str, default='vai_for_resuneta',
                        choices=['potsdam', 'segbase', 'vai_for_resuneta'],
                        help='dataset name (default: pascal_voc)')
    ###########
    # parser.add_argument('--checkpoint',
    #                     default="/home/caoyiwen/slns/XLPR_Semantic_Segmentation-master/runs/models/pspnet_potsdam_best_model.pth",
    #                     help='Directory for saving checkpoint models')
    ###########

    # training hyper params
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--seed', type=int, default=1024, help='Default 1024')
    parser.add_argument('--use_ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'ASGD', 'Adam', 'Rprop', 'Adagrad',
                                 'Adadelta', 'RMSprop', 'Adamax', 'SparseAdam', 'LBFGS'],
                        help='Default gesture')
    parser.add_argument('--lr_scheduler', type=str, default='WarmupPolyLR',
                        choices=['MultiStepLR', 'StepLR', 'ExponentialLR', 'WarmupPolyLR'],
                        help='Default gesture')
    parser.add_argument('--loss', type=str, default=['Dice_Loss', 'Cross_Entorpy_Loss'],
                        choices=['Cross_Entorpy_Loss', 'Dice_Loss', 'Lovasz_Softmax_Loss', ],
                        help='Default gesture')
    parser.add_argument('--apex', default=False, help='Turn on the mixed accuracy training Apex')

    # checkpoint and log
    parser.add_argument('--save_dir', default='./runs/models/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_epoch', type=int, default=1,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log_dir', default='./runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_iter', type=int, default=50,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val_epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--device', type=str, default='0', help='The ids of GPU to be used')
    args = parser.parse_args()  # 解析
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # dataset and dataloader
        train_dataset = get_segmentation_dataset(args.dataset, split='train')
        val_dataset = get_segmentation_dataset(args.dataset, split='val')

        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('train dataset len: {}'.format(len(self.train_loader.dataset)))
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('val dataset len: {}'.format(len(self.val_loader.dataset)))
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch
        print('load dateset finished')
        # create network 加载模型和数据集
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset).to(self.device)

        self.opt = get_optimizer(type=args.optimizer, model=self.model, lr=args.lr, wd=args.weight_decay)
        print("optimizer is ", self.opt)
        # learning rate scheduler
        self.sch = get_lr_scheduler(type=args.lr_scheduler, opt=self.opt, max_epoch=args.epochs,
                                    iter_epoch=len(self.train_loader))
        # loss function
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1,
                                               nclass=train_dataset.num_class).to(self.device)
        print(self.criterion)
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        # Apex
        if args.apex:
            self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")
        if torch.cuda.is_available():
            self.model.cuda()
            self.criterion.cuda()
        # 多卡并行模型初始化,用不着那么多显存
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.model = nn.DataParallel(self.model).cuda()
        self.best_pred = 0.0

    def train(self):
        start_time = time.time()
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        val_per_iters = self.args.val_epoch * self.args.iters_per_epoch
        logger.info(
            'Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(self.args.epochs, args.max_iters))
        # self.model = torch.load(args.checkpoint)
        # print("success!")
        self.model.train()
        for epoch in range(self.args.epochs):
            iterations = len(self.train_loader) * epoch
            for iteration, (images, targets, _) in enumerate(self.train_loader):
                iteration = iteration + 1
                self.sch.step()
                self.opt.zero_grad()
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                if args.apex:
                    with amp.scale_loss(losses, self.opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    losses.backward()
                self.opt.step()
                eta_seconds = ((time.time() - start_time) / (iteration + iterations)) * (
                            args.max_iters - (iteration + iterations))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if iteration % self.args.log_iter == 0:
                    logger.info(
                        "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            int(iteration + iterations), args.max_iters, self.opt.param_groups[0]['lr'], losses.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string))

                if iteration % save_per_iters == 0:
                    save_checkpoint(self.model, self.args, is_best=False)

                if iteration % val_per_iters == 0:
                    self.validation()
                    self.model.train()

            save_checkpoint(self.model, self.args, is_best=False)
            total_training_time = time.time() - start_time
            total_training_str = str(datetime.timedelta(seconds=total_training_time))
            logger.info(
                "Total training time: {} ({:.4f}s / it)".format(
                    total_training_str, total_training_time / args.max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, dice, mIoU = self.metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, dice: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, dice, mIoU))

        new_pred = dice
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    time1_str = datetime.datetime.strftime(datetime.datetime.now(), '%F')
    filename = '{}_{}_{}.pth'.format(args.model, args.dataset, time1_str)
    filename = os.path.join(directory, filename)
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model_{}.pth'.format(args.model, args.dataset, time1_str)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()  # 提前设定的值
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    logger = setup_logger("semantic_segmentation", args.log_dir, filename='{}_{}_log.txt'.format(
        args.model, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
