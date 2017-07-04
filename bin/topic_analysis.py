#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 14:13
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


def count_topic(topic_mat):
    topic_cnt = {}
    for vec in topic_mat:
        for tid in vec:
            topic_cnt[tid] = topic_cnt.get(tid, 0.) + 1.
    return topic_cnt




