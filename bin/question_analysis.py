#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/4 11:24
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from data_utils import load_question_set
import ConfigParser
import sys
from utils import LogUtil


def length_analysis(que_list):
    max_len = 0
    min_len = sys.maxint
    ave_len = 0

    for vec in que_list:
        vec_len = len(vec)
        max_len = max(max_len, vec_len)
        min_len = min(min_len, vec_len)
        ave_len += vec_len
    ave_len /= (1. * len(que_list))
    LogUtil.log('INFO', 'max_len=%d' % max_len)
    LogUtil.log('INFO', 'min_len=%d' % min_len)
    LogUtil.log('INFO', 'ave_len=%d' % ave_len)


def all_length_analysis(config):
    question_set_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt.small'
    qid_list, tc_list, tw_list, dc_list, dw_list = load_question_set(question_set_fp)

    LogUtil.log('INFO', 'analysis length of title char:')
    length_analysis(tc_list)
    LogUtil.log('INFO', 'analysis length of title word:')
    length_analysis(tw_list)
    LogUtil.log('INFO', 'analysis length of document char:')
    length_analysis(dc_list)
    LogUtil.log('INFO', 'analysis length of document word:')
    length_analysis(dw_list)


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    all_length_analysis(config)