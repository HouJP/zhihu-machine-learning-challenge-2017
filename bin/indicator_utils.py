#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 13:49
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import logging


def pmc(mat):
    """
    Calculation of PMC (Percentage of  documents belonging to More than one Category)
    :param mat:
    :return:
    """
    cnt = 0.
    for vec in mat:
        if 1 < len(vec):
            cnt += 1
    ind_val = cnt / len(mat)
    logging.info('PMC(%s)' % ind_val)
    return ind_val


def anl(mat):
    """
    Calculation of ANL (Average Number of Labels for each document)
    :param mat:
    :return:
    """
    cnt = 0.
    for vec in mat:
        cnt += len(vec)
    ind_val = cnt / len(mat)
    logging.info('ANL(%s)' % ind_val)
    return ind_val

