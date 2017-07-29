#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 10:11
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
from ..text_cnn.data_helpers import parse_feature_vec
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file


def vote(config, argv):
    dataset_name = argv[0]
    dataset_pt = config.get('DIRECTORY', 'dataset_pt')
    index_pt = config.get('DIRECTORY', 'index_pt')
    class_num = config.getint('DATA_ATTRIBUTE', 'class_num')
    topk = config.getint('RANK', 'topk')
    vote_features = config.get('RANK', 'vote_features').split()
    vote_features_f = [open('%s/%s.%s.csv' % (dataset_pt, fn, dataset_name), 'r') for fn in vote_features]
    topk_class_index_f = open('%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), dataset_name), 'w')

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

        #print aggregator
        topk_ids = [kv[0] for kv in sorted(enumerate(aggregator), key=lambda x: x[1], reverse=True)[:topk]]
        topk_class_index_f.write('%s\n' % ' '.join([str(n) for n in topk_ids]))

    topk_class_index_f.close()
    for f in vote_features_f:
        f.close()


def analyze_vote(config, argv):
    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
    valid_index_off = [num - 1 for num in valid_index_off]

    # load valid dataset
    valid_label_id = load_labels_from_file(config, 'offline', valid_index_off)

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    topk_class_index_fp = '%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), 'offline')
    topk_label_id = DataUtil.load_matrix(topk_class_index_fp, 'int')

    total_labels = 0
    right_labels = 0

    topk = config.getint('RANK', 'topk')
    class_num = config.getint('DATA_ATTRIBUTE', 'class_num')
    assert len(valid_label_id) == len(topk_label_id)
    for line_id in range(len(topk_label_id)):
        for class_id in range(class_num):
            if 1 == valid_label_id[line_id][class_id]:
                total_labels += 1
        for lid in topk_label_id[line_id]:
            if 1 == valid_label_id[line_id][lid]:
                right_labels += 1
    LogUtil.log('INFO', 'topk=%d, recall=%s%%' % (topk, str(100. * right_labels / total_labels)))


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
