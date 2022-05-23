import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def tanimoto_loss(label, pred):
    square = torch.square(pred)
    sum_square = torch.sum(square)
    product = torch.multiply(pred, label)
    sum_product = torch.sum(product)
    denomintor = torch.subtract(torch.add(sum_square, 1), sum_product)
    loss = torch.divide(sum_product, denomintor)
    loss = torch.reduce_mean(loss)
    return 1.0 - loss



ALPHA = 0.5
BETA = 0.5


def dice_loss(pred, target, epsilon=1e-7, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.sigmoid(pred)
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss