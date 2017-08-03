#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/3 00:40
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from text_cnn.data_helpers import load_raw_line_from_file
from data_utils import parse_question_set
from utils import DataUtil
from featwheel.feature import Feature
import sys
import ConfigParser


def generate(config, argv):
    data_name = argv[0]

    # load data set
    if 'offline' == data_name:
        # load offline valid dataset index
        valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                      config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
        valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
        valid_index_off = [num - 1 for num in valid_index_off]

        source_file_path = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
        source_data = load_raw_line_from_file(config, source_file_path, valid_index_off)
    elif 'online' == data_name:
        source_file_path = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
        source_data = open(source_file_path, 'r').readlines()
    else:
        source_data = None

    feature_file_path = '%s/instance_fs_length.%s.smat'
    feature_file = open(feature_file_path, 'w')

    feature_file.write('%d %d\n' % (len(source_data, 4)))
    for line in feature_file:
        qid, tc, tw, dc, dw = parse_question_set(line)
        feature = list()
        feature.append(len(tc))
        feature.append(len(tw))
        feature.append(len(dc))
        feature.append(len(dw))
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

