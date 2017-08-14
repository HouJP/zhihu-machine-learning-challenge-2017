#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/14 22:34
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
import math
from utils import LogUtil
from data_utils import load_topic_info, load_question_set, parse_question_set
import json
from utils import DataUtil
from text_cnn.data_helpers import load_raw_line_from_file
from featwheel.feature import Feature


def generate(config, argv):
    data_name = argv[0]

    word_idf_fp = '%s/words.idf' % config.get('DIRECTORY', 'devel_pt')
    with open(word_idf_fp, 'r') as word_idf_f:
        word_idf = json.load(word_idf_f)
    LogUtil.log("INFO", "load word_idf done, len(word_idf)=%d" % len(word_idf))

    char_idf_fp = '%s/chars.idf' % config.get('DIRECTORY', 'devel_pt')
    with open(char_idf_fp, 'r') as char_idf_f:
        char_idf = json.load(char_idf_f)
    LogUtil.log("INFO", "load char_idf done, len(char_idf)=%d" % len(char_idf))

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

    idf_sum_feature_file_path = '%s/instance_fs_idf_sum.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'), data_name)
    feature_file = open(idf_sum_feature_file_path, 'w')

    feature_file.write('%d %d\n' % (len(source_data), 4))
    for line in source_data:
        qid, tc, tw, dc, dw = parse_question_set(line)
        feature = list()
        feature.append(sum([char_idf[char] for char in tc if len(char) > 0]))
        feature.append(sum([word_idf[word] for word in tw if len(word) > 0]))

        feature.append(sum([char_idf[char] for char in dc if len(char) > 0]))
        feature.append(sum([word_idf[word] for word in dw if len(word) > 0]))

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