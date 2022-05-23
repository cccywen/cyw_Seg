#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 12:14
# @Author:Jianyuan Hong
# @File:model_zoo.py
# @Software:PyCharm

"""Model store which handles pretrained models """
from .unet import *
from .resuneta import *
from .resuneta_py import *
from .pspnet import *
__all__ = ['get_segmentation_model']

def get_segmentation_model(model, **kwargs):
    models = {
        'unet': get_unet,
        'resuneta': get_resuneta,
        'resuneta_py': get_resuneta_py,
        # 'pfsegnet': get_pfsegnet,
        'pspnet': get_pspnet,
    }
    return models[model](**kwargs)



