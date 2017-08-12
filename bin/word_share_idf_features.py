#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/18 09:29
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


def load_topic_info_sort(config):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

    # load hash table of label
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    tc_sort = [[]] * 1999
    tw_sort = [[]] * 1999
    dc_sort = [[]] * 1999
    dw_sort = [[]] * 1999

    for line_id in range(1999):
        tid = int(label2id[tid_list[line_id]])
        tc_sort[tid] = tc_list[line_id]
        tw_sort[tid] = tw_list[line_id]
        dc_sort[tid] = dc_list[line_id]
        dw_sort[tid] = dw_list[line_id]

    return tc_sort, tw_sort, dc_sort, dw_sort


def generate(config, argv):
    data_name = argv[0]

    word_idf_fp = '%s/words.idf' % config.get('DIRECTORY', 'devel_pt')
    with open(word_idf_fp, 'r') as word_idf_f:
        word_idf = json.load(word_idf_f)
    LogUtil.log("INFO", "load word_idf done, len(word_idf)=%d" % len(word_idf))

    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    topic_tc, topic_tw, topic_dc, topic_dw = load_topic_info_sort(config)
    topic_tc = [set(tc) for tc in topic_tc]
    topic_tw = [set(tw) for tw in topic_tw]
    topic_dc = [set(dc) for dc in topic_dc]
    topic_dw = [set(dw) for dw in topic_dw]

    if 'offline' == data_name:
        source_file_path = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
        source_data = load_raw_line_from_file(config, source_file_path, valid_index)
    elif 'online' == data_name:
        source_file_path = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
        source_data = open(source_file_path, 'r').readlines()
    else:
        source_data = None

    pair_tws_idf_feature_file_path = '%s/pair_title_word_share_idf.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), data_name)
    pair_tws_idf_feature_file = open(pair_tws_idf_feature_file_path, 'w')

    pair_dws_idf_feature_fp = '%s/pair_content_word_share_idf.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), data_name)
    pair_dws_idf_feature_f = open(pair_dws_idf_feature_fp, 'w')

    # feature_file.write('%d %d\n' % (len(source_data), 4))
    line_id = 0
    for line in source_data:
        qid, tc, tw, dc, dw = parse_question_set(line)
        tw_features = list()
        for tid in range(1999):
            agg = 0.
            for word in tw:
                if word in topic_tw[tid] and len(word):
                    agg += word_idf[word]
            tw_features.append(agg)
        pair_tws_idf_feature_file.write(','.join([str(num) for num in tw_features]))

        dw_features = list()
        for tid in range(1999):
            agg = 0.
            for word in dw:
                if word in topic_dw[tid] and len(word):
                    agg += word_idf[word]
            dw_features.append(agg)
        pair_dws_idf_feature_f.write(','.join([str(num) for num in dw_features]))

        if 0 == line_id % 10000:
            LogUtil.log('INFO', str(line_id))
        line_id += 1

    pair_tws_idf_feature_file.close()
    pair_dws_idf_feature_f.close()


def generate_idf(config, argv):
    question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    question_online_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'

    qid_off, tc_off, tw_off, dc_off, dw_off = load_question_set(question_offline_fp)
    qid_on, tc_on, tw_on, dc_on, dw_on = load_question_set(question_online_fp)

    word_idf = dict()

    for line_id in range(len(qid_off)):
        words = set(tw_off[line_id] + dw_off[line_id])
        for word in words:
            word_idf[word] = word_idf.get(word, 0) + 1
        if line_id % 10000 == 0:
            print '%s %d' % ('offline word', line_id)

    for line_id in range(len(qid_on)):
        words = set(tw_on[line_id] + dw_on[line_id])
        for word in words:
            word_idf[word] = word_idf.get(word, 0) + 1
        if line_id % 10000 == 0:
            print '%s %d' % ('online word', line_id)

    num_docs = len(qid_off) + len(qid_on)
    for word in word_idf:
        word_idf[word] = math.log(num_docs / (word_idf[word] + 1.)) / math.log(2.)

    word_idf_fp = '%s/words.idf' % config.get('DIRECTORY', 'devel_pt')
    with open(word_idf_fp, 'w') as word_idf_f:
        json.dump(word_idf, word_idf_f)

    LogUtil.log("INFO", "word_idf calculation done, len(word_idf)=%d" % len(word_idf))

    char_idf = dict()

    for line_id in range(len(qid_off)):
        chars = set(tc_off[line_id] + dc_off[line_id])
        for char in chars:
            char_idf[char] = char_idf.get(char, 0) + 1
        if line_id % 10000 == 0:
            print '%s %d' % ('offline char', line_id)

    for line_id in range(len(qid_on)):
        chars = set(tc_on[line_id] + dc_on[line_id])
        for char in chars:
            char_idf[char] = char_idf.get(char, 0) + 1
        if line_id % 10000 == 0:
            print '%s %d' % ('online char', line_id)

    for char in char_idf:
        char_idf[char] = math.log(num_docs / (char_idf[char] + 1.)) / math.log(2.)

    char_idf_fp = '%s/chars.idf' % config.get('DIRECTORY', 'devel_pt')
    with open(char_idf_fp, 'w') as char_idf_f:
        json.dump(char_idf, char_idf_f)

    LogUtil.log("INFO", "char_idf calculation done, len(char_idf)=%d" % len(char_idf))



def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)
