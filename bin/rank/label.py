#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 23:17
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file


def generate(config, argv):
    data_name = 'offline'
    rank_id = config.get('RANK', 'rank_id')
    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    topk_class_index_fp = '%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), data_name)
    topk_label_id = DataUtil.load_matrix(topk_class_index_fp, 'int')
    LogUtil.log('INFO', 'topk_class_index_fp=%s' % topk_class_index_fp)

    # load labels
    feature_name = 'labels'
    LogUtil.log('INFO', 'feature_name=%s' % feature_name)
    rank_features_fp = '%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, rank_id, data_name)
    LogUtil.log('INFO', 'rank_features_fp=%s' % rank_features_fp)
    rank_features_f = open(rank_features_fp, 'w')
    features = load_labels_from_file(config, data_name, valid_index)

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
