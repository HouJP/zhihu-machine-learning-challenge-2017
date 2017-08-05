#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 23:17
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
import hashlib
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file


def generate_offline(config, argv):
    data_name = 'offline'

    # load labels
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]
    labels = load_labels_from_file(config, data_name, valid_index)

    # load vote_k ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name)
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    assert len(labels) == len(vote_k_label)

    featwheel_label_file_path = '%s/featwheel_vote_%d_%s.%s.label' % (
        config.get('DIRECTORY', 'label_pt'),
        vote_k,
        vote_k_label_file_name,
        data_name)
    LogUtil.log('INFO', 'featwheel_label_file_path=%s' % featwheel_label_file_path)
    featwheel_feature_file = open(featwheel_label_file_path, 'w')

    for line_id in range(len(labels)):
        for label_id in vote_k_label[line_id]:
            featwheel_feature_file.write('%d\n' % labels[line_id][label_id])

    featwheel_feature_file.close()


def generate_online(config, argv):
    data_name = 'online'

    # load vote_k ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name)
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    featwheel_label_file_path = '%s/featwheel_vote_%d_%s.%s.label' % (
        config.get('DIRECTORY', 'label_pt'),
        vote_k,
        vote_k_label_file_name,
        data_name)
    LogUtil.log('INFO', 'featwheel_feature_file_path=%s' % featwheel_label_file_path)
    featwheel_label_file = open(featwheel_label_file_path, 'w')

    for line_id in range(len(vote_k_label)):
        for label_id in vote_k_label[line_id]:
            featwheel_label_file.write('0\n')

    featwheel_label_file.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
