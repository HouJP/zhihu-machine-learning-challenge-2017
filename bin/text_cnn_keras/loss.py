#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 15:59
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from keras import backend


def binary_crossentropy_sum(y_true, y_pred):
    return backend.sum(backend.binary_crossentropy(y_pred, y_true), axis=-1)
