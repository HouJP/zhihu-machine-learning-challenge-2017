#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 14:13
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from text_cnn.data_helpers import parse_lid_vec
import sys
import ConfigParser
from pylab import *
import numpy as np


def count_topic(topic_mat):
    topic_cnt = {}
    for vec in topic_mat:
        for tid in vec:
            topic_cnt[tid] = topic_cnt.get(tid, 0.) + 1.
    return topic_cnt


def plot_lid_num(config):
    lid_fp = '%s/label_id.offline.csv' % config.get('DIRECTORY', 'dataset_pt')
    lid_f = open(lid_fp, 'r')

    num = []
    tol = 0
    spe_lid = 106
    ind = 0
    for line in lid_f:
        vec = parse_lid_vec(line, 1999)
        if 0 == ind % 10000:
            num.append(tol)
            tol = 0
        if 1 == vec[spe_lid]:
            tol += 1
        ind += 1

    print num
    plot(range(len(num))[:100], num[:100])
    show()
    lid_f.close()


def main():
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    func = sys.argv[2]

    eval(func)(config)


if __name__ == '__main__':
    # _test()
    main()

