#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/18 09:29
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
import math
from utils import LogUtil
from data_utils import load_topic_info, load_question_set
import json


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

    for line_id in range(len(qid_on)):
        words = set(tw_on[line_id] + dw_on[line_id])
        for word in words:
            word_idf[word] = word_idf.get(word, 0) + 1

    num_docs = len(qid_off) + len(qid_on)
    for word in word_idf:
        word_idf[word] = math.log(num_docs / (word_idf[word] + 1.)) / math.log(2.)

    word_idf_fp = '%s/words.idf' % config.get('DIRECTORY', 'devel_pt')
    json.dump(word_idf, word_idf_fp)

    LogUtil.log("INFO", "word_idf calculation done, len(word_idf)=%d" % len(word_idf))

    char_idf = dict()

    for line_id in range(len(qid_off)):
        chars = set(tc_off[line_id] + dc_off[line_id])
        for char in chars:
            char_idf[char] = char_idf.get(char, 0) + 1

    for line_id in range(len(qid_on)):
        chars = set(tc_on[line_id] + dc_on[line_id])
        for char in chars:
            char_idf[char] = char_idf.get(char, 0) + 1

    for char in char_idf:
        char_idf[char] = math.log(num_docs / (char_idf[char] + 1.)) / math.log(2.)

    char_idf_fp = '%s/chars.idf' % config.get('DIRECTORY', 'devel_pt')
    json.dump(char_idf, char_idf_fp)

    LogUtil.log("INFO", "char_idf calculation done, len(char_idf)=%d" % len(char_idf))



def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)
