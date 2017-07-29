#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 10:11
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
from ..text_cnn.data_helpers import parse_feature_vec


def vote(config, dataset_name):
    dataset_pt = config.get('DIRECTORY', 'dataset_pt')
    index_pt = config.get('DIRECTORY', 'index_pt')
    class_num = config.getint('DATA_ATTRIBUTE', 'class_num')
    topk = config.getint('RANK', 'topk')
    vote_features = config.get('RANK', 'vote_features').split()
    vote_features_f = [open('%s/%s.%s' % (dataset_pt, fn, dataset_name), 'r') for fn in vote_features]
    topk_class_index_f = open('%s/%s.%s' % (index_pt, config.get('RANK', 'topk_class_index'), dataset_name), 'w')

    while True:
        aggregator = [0.] * class_num
        eof = False
        for f in vote_features_f:
            line = f.readline()
            if '' == line:
                eof = True
                break
            vec = parse_feature_vec(line)
            for i in range(class_num):
                aggregator[i] += vec[i]
        if eof:
            break

        print aggregator
        topk_ids = [kv[0] for kv in sorted(enumerate(aggregator), key=lambda x: x[1], reverse=True)[:topk]]
        topk_class_index_f.write('%s\n' % ' '.join([str(n) for n in topk_ids]))

    topk_class_index_f.close()
    for f in vote_features_f:
        f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    dataset_name = sys.argv[2]

    vote(config, dataset_name)
