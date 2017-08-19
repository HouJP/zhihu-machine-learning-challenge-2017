#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 10:11
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import os
import ConfigParser
import hashlib
from ..text_cnn.data_helpers import parse_feature_vec
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file


def find_vote_feature_file(model_name, data_name):
    RootDir = ['/mnt/disk2/xinyu/data/dataset/', '/home/xinyu/zhihu_preds/from_124/']
    RootDir.append('/mnt/disk2/xinyu/niuox_data/RCNN/')

    model_name = model_name.strip().strip('\n')
    FileTemp = ''
    if len(model_name) != 0 and model_name[0] != '#':
        for rd in RootDir:
            if os.path.isfile('%s/%s.%s.csv' % (rd, model_name, data_name)):
                FileTemp = '%s/%s.%s.csv' % (rd, model_name, data_name)
                break
            elif os.path.isfile('%s/%s.%s.preds' % (rd, model_name, data_name)):
                FileTemp = '%s/%s.%s.preds' % (rd, model_name, data_name)
                break
    if '' == FileTemp:
        LogUtil.log('INFO', 'can\'t find %s' % model_name)

    return FileTemp


def vote(config, argv):
    data_name = argv[0]

    dataset_pt = config.get('DIRECTORY', 'dataset_pt')
    index_pt = config.get('DIRECTORY', 'index_pt')
    class_num = config.getint('DATA_ATTRIBUTE', 'class_num')
    vote_k = config.getint('RANK', 'vote_k')

    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_feature_files = [find_vote_feature_file(fn, data_name) for fn in vote_feature_names]
    vote_feature_files = [open(fn, 'r') for fn in vote_feature_files if len(fn) > 0]

    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k_label_file = open('%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name), 'w')

    LogUtil.log('INFO', 'vote_feature_names=%s' % str(vote_feature_names))
    LogUtil.log('INFO', 'vote_%d_label_file_name=%s' % (vote_k, str('%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name))))

    while True:
        aggregator = [0.] * class_num
        eof = False
        for f in vote_feature_files:
            line = f.readline()
            if '' == line:
                eof = True
                break
            vec = parse_feature_vec(line)
            for i in range(class_num):
                aggregator[i] += vec[i]
        if eof:
            break

        vote_k_ids = [kv[0] for kv in sorted(enumerate(aggregator), key=lambda x: x[1], reverse=True)[:vote_k]]
        vote_k_label_file.write('%s\n' % ' '.join([str(n) for n in vote_k_ids]))

    vote_k_label_file.close()
    for f in vote_feature_files:
        f.close()
    analyze_vote(config, [])


def analyze_vote(config, argv):
    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
    valid_index_off = [num - 1 for num in valid_index_off]

    # load valid dataset
    valid_label_id = load_labels_from_file(config, 'offline', valid_index_off)

    # load topk ids
    vote_k = config.getint('RANK', 'vote_k')
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    total_labels = 0
    right_labels = 0

    vote_k = config.getint('RANK', 'vote_k')
    class_num = config.getint('DATA_ATTRIBUTE', 'class_num')
    assert len(valid_label_id) == len(vote_k_label)
    for line_id in range(len(vote_k_label)):
        for class_id in range(class_num):
            if 1 == valid_label_id[line_id][class_id]:
                total_labels += 1
        for lid in vote_k_label[line_id]:
            if 1 == valid_label_id[line_id][lid]:
                right_labels += 1

    LogUtil.log('INFO', 'vote_k=%d, recall=%s%%' % (vote_k, str(100. * right_labels / total_labels)))


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
