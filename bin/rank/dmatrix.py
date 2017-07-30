#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/30 00:04
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import re
import ConfigParser

from ..text_cnn.data_helpers import parse_feature_vec


def generate(config, argv):
    rank_id = config.get('RANK', 'rank_id')
    data_name = argv[0]
    # load rank features
    feature_names = config.get('RANK', 'rank_features').split()
    feature_files = [open('%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), fn, data_name), 'r') for fn in feature_names]

    # load rank labels
    label_file = open('%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'labels', rank_id, data_name), 'r')

    dmatrix_file = open('%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'dmatrix', rank_id, data_name),
                        'w')
    group_file = open('%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'group', rank_id, data_name),
                      'w')

    while True:
        features = list()
        end_while = False
        for feature_file in feature_files:
            line = feature_file.readline()
            if '' == line:
                end_while = True
                break
            features.append(parse_feature_vec(line))
        if end_while:
            break

        line = label_file.readline()
        label = [int(num) for num in re.split(' |,', line.strip('\n'))]

        for i in range(len(label)):
            vec = list()
            for j in range(len(features)):
                vec.append(features[j][i])
            s = ' '.join(['%d:%s' % (kv[0], kv[1]) for kv in enumerate(vec)])
            dmatrix_file.write('%d %s\n' % (label[i], s))
        group_file.write('%d\n' % len(label))

    for f in feature_files:
        f.close()
    label_file.close()
    dmatrix_file.close()
    group_file.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)