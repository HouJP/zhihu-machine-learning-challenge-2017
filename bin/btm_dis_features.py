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
from text_cnn.data_helpers import parse_feature_vec, load_features_from_file
from utils import DataUtil, LogUtil


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
    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load topic btm vec
    topic_btm_vec = load_topic_btm_vec(config)

    # offline / online
    data_name = argv[0]

    dis_func_names = ["cosine",
                      "cityblock",
                      "jaccard",
                      "canberra",
                      "euclidean",
                      "minkowski",
                      "braycurtis"]

    btm_dis_feature_fn = ['fs_btm_dis_%s' % dis_func_name for dis_func_name in dis_func_names]
    btm_dis_feature_f = [open('%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'),
                                                fn,
                                                data_name), 'w') for fn in btm_dis_feature_fn]

    if 'offline' == data_name:
        btm_tw_cw_features = load_features_from_file(config, 'fs_btm_tw_cw', data_name, valid_index)
        LogUtil.log('INFO', 'load_features_from_file, len=%d' % len(btm_tw_cw_features))
        for line_id in range(len(btm_tw_cw_features)):
            doc_vec = btm_tw_cw_features[line_id]
            for dis_id in range(len(dis_func_names)):
                vec = [0.] * 1999
                for topic_id in range(1999):
                    topic_vec = topic_btm_vec[topic_id]
                    if 'minkowski' == dis_func_names[dis_id]:
                        vec[topic_id] = eval(dis_func_names[dis_id])(doc_vec, topic_vec, 3)
                    else:
                        vec[topic_id] = eval(dis_func_names[dis_id])(doc_vec, topic_vec)
                btm_dis_feature_f[dis_id].write('%s\n' % ','.join([str(num) for num in vec]))
    else:
        btm_vec_fp = '%s/fs_btm_tw_cw.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), data_name)
        btm_vec_f = open(btm_vec_fp, 'r')
        for line in btm_vec_f:
            doc_vec = np.nan_to_num(parse_feature_vec(line))
            for dis_id in range(len(dis_func_names)):
                vec = [0.] * 1999
                for topic_id in range(1999):
                    topic_vec = topic_btm_vec[topic_id]
                    if 'minkowski' == dis_func_names[dis_id]:
                        vec[topic_id] = eval(dis_func_names[dis_id])(doc_vec, topic_vec, 3)
                    else:
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
