#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 8/4/17 12:21 PM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
import json
from data_utils import load_topic_info
from featwheel.feature import Feature


def generate(config, argv):
    # load topic info
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

    # load hash table of label
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    feature_file_path = '%s/topic_fs_length.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'), 'all')
    feature_file = open(feature_file_path, 'w')
    features = [0] * len(tid_list)

    for line_id in range(len(tid_list)):
        feature = list()
        feature.append(len(father_list[line_id]))
        feature.append(len(tc_list[line_id]))
        feature.append(len(tw_list[line_id]))
        feature.append(len(dc_list[line_id]))
        feature.append(len(dw_list[line_id]))

        label_id = int(label2id[tid_list[line_id]])
        features[label_id] = feature

    for feature in features:
        Feature.save_feature(feature, feature_file)

    feature_file.close()


def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)

