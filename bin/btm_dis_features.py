#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 8/2/17 8:39 PM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com


from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import numpy as np
import re
import json
import sys
import ConfigParser
from data_utils import load_topic_info
from text_cnn.data_helpers import parse_feature_vec


def load_topic_btm_vec(config):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

    # load hash table of label
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    btm_topic_vec_fp = '%s/fs_btm_tw_cw.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'topic')
    btm_topic_vec_f = open(btm_topic_vec_fp, 'r')

    topic_btm_vecs = [0.] * 1999

    line_id = 0
    for line in btm_topic_vec_f:
        vec = np.nan_to_num(parse_feature_vec(line))
        topic_btm_vecs[int(label2id[tid_list[line_id]])] = vec
        line_id += 1

    return topic_btm_vecs


def generate(config, argv):
    # load topic btm vec
    topic_btm_vec = load_topic_btm_vec(config)

    # offline / online
    data_name = argv[0]

    btm_vec_fp = '%s/fs_btm_tw_cw.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), data_name)
    btm_vec_f = open(btm_vec_fp, 'r')

    dis_func_names = ["cosine",
                      "cityblock",
                      "jaccard",
                      "canberra",
                      "euclidean",
                      "euclidean",
                      "euclidean"]

    btm_dis_feature_fn = ['fs_btm_dis_%s' % dis_func_name for dis_func_name in dis_func_names]
    btm_dis_feature_f = [open('%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'),
                                                fn,
                                                data_name), 'w') for fn in btm_dis_feature_fn]

    for line in btm_vec_f:
        doc_vec = np.nan_to_num(parse_feature_vec(line))
        for dis_id in range(len(dis_func_names)):
            vec = [0.] * 1999
            for topic_id in range(1999):
                topic_vec = topic_btm_vec[topic_id]
                vec[topic_id] = eval(dis_func_names[dis_id])(doc_vec, topic_vec)
            btm_dis_feature_f[dis_id].write('%s\n' % ','.join([str(num) for num in vec]))

    for f in btm_dis_feature_f:
        f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
