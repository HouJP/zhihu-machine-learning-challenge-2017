#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 14:13
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
from data_utils import load_topic_info
from utils import LogUtil
from question_analysis import length_analysis


def count_topic(topic_mat):
    topic_cnt = {}
    for vec in topic_mat:
        for tid in vec:
            topic_cnt[tid] = topic_cnt.get(tid, 0.) + 1.
    return topic_cnt


def all_length_analysis(config):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

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