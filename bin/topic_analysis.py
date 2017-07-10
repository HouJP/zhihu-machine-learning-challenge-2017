#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 14:13
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from text_cnn.data_helpers import load_lid
import sys
import ConfigParser


def count_topic(topic_mat):
    topic_cnt = {}
    for vec in topic_mat:
        for tid in vec:
            topic_cnt[tid] = topic_cnt.get(tid, 0.) + 1.
    return topic_cnt


def plot_lid_num(config):
    lid_fp = '%s/label_id.offline.csv' % config.get('DIRECTORY', 'dataset_pt')
    lid = load_lid(lid_fp, 1999)

    num = []
    tol = 0
    spe_lid = 99
    for i in range(len(lid)):
        if 0 == i % 10000:
            num.append(tol)
        if 1 == lid[i][spe_lid]:
            tol += 1

    print num


def main():
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    func = sys.argv[2]

    eval(func)(config)


if __name__ == '__main__':
    # _test()
    main()

