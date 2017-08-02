#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 14:59
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
from ..utils import DataUtil, LogUtil
from os.path import isfile
from ..text_cnn.data_helpers import load_features_from_file


def generate(config, argv):
    data_name = argv[0]
    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    topk_class_index_fp = '%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), data_name)
    topk_label_id = DataUtil.load_matrix(topk_class_index_fp, 'int')

    # load rank features
    feature_names = config.get('RANK', 'rank_features').split()
    for feature_name in feature_names:
        LogUtil.log('INFO', 'feature_name=%s' % feature_name)
        rank_features_fp = '%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, data_name)
        LogUtil.log('INFO', 'rank_features_fp=%s' % rank_features_fp)
        has_rank_features = isfile('%s' % rank_features_fp)
        if has_rank_features:
            LogUtil.log('INFO', 'has rank features, jump')
            continue
        rank_features_f = open(rank_features_fp, 'w')
        if 'offline' == data_name and 0 == feature_name.count('vote_'):
            LogUtil.log('INFO', 'load_features_from_file')
            features = load_features_from_file(config, feature_name, data_name, valid_index)
        else:
            LogUtil.log('INFO', 'load_matrix')
            features = DataUtil.load_matrix('%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, data_name), 'float')

        assert len(topk_label_id) == len(features)
        for line_id in range(len(topk_label_id)):
            rank_features = [str(features[line_id][lid]) for lid in topk_label_id[line_id]]
            rank_features_f.write('%s\n' % ','.join(rank_features))

        rank_features_f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)