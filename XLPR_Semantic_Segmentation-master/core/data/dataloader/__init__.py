#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 10:08
# @Author:Jianyuan Hong
# @File:__init__.py.py
# @Software:PyCharm

"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .FUSeg import FootUlcerSegmentation
from .potsdam import PotsdamSegmentation
from .vai_for_resuneta import VaiSegmentation


datasets = {
    'fuseg': FootUlcerSegmentation,
    'potsdam': PotsdamSegmentation,
    'vai_for_resuneta': VaiSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
