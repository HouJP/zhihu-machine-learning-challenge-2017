#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 23:11
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import data_utils
import logging
import sys


def save_question_topic_info(cf):
    q_train_set = cf.get('DEFAULT', 'source_pt') + '/question_train_set.txt'
    (qid_train_list, tc_train_list, tw_train_list, dc_train_list, dw_train_list) = data_utils.load_question_set(q_train_set)

    q_eval_set = cf.get('DEFAULT', 'source_pt') + '/question_eval_set.txt'
    (qid_eval_list, tc_eval_list, tw_eval_list, dc_eval_list, dw_eval_list) = data_utils.load_question_set(q_eval_set)

    q_topic_set = cf.get('DEFAULT', 'source_pt') + '/topic_info.txt'
    (tid_topic_list, father_topic_list, tc_topic_list, tw_topic_list, dc_topic_list, dw_topic_list) = data_utils.load_topic_info(q_topic_set)

    btm_qt_info_fp = cf.get('DEFAULT', 'devel_pt') + '/btm_qt_info.txt'
    f = open(btm_qt_info_fp, 'w')
    for i in range(len(qid_train_list)):
        s = ' '.join((tw_train_list[i] + dw_train_list[i])) + '\n'
        if 0 == s.strip():
            logging.warn('question_train_set.txt has no content at line#%d' % i)
            s = 'empty\n'
        f.write(s)
    for i in range(len(qid_eval_list)):
        s = ' '.join((tw_eval_list[i] + dw_eval_list[i])) + '\n'
        if 0 == s.strip():
            logging.warn('question_eval_set.txt has no content at line#%d' % i)
            s = 'empty\n'
        f.write(s)
    for i in range(len(tid_topic_list)):
        s = ' '.join((tw_topic_list[i] + dw_topic_list[i])) + '\n'
        if 0 == s.strip():
            logging.warn('topic_info.txt has no content at line#%d' % i)
            s = 'empty\n'
        f.write(s)
    f.close()


def btm2standard_format(config, argv):
    tmp_f = open('%s/btm_embedding.tmp' % config.get('DIRECTORY', 'embedding_pt'), 'r')
    btm_f = open('%s/btm_embedding.txt' % config.get('DIRECTORY', 'embedding_pt'), 'w')

    line_num = 3219326
    btm_f.write('%d 100\n' % line_num)
    ind = 0
    for line in tmp_f:
        btm_f.write('%d %s' % (ind, line))
        ind += 1

    tmp_f.close()
    btm_f.close()


def generate_btm_csv(config, argv):
    btm_off_f = open('%s/btm.offline.csv' % config.get('DIRECTORY', 'dataset_pt'), 'w')
    btm_on_f = open('%s/btm.online.csv' % config.get('DIRECTORY', 'dataset_pt'), 'w')

    off_num = 2999967
    on_num = 217360
    for i in range(off_num):
        btm_off_f.write('%d\n' % i)
    for i in range(off_num, off_num + on_num):
        btm_on_f.write('%d\n' % i)

    btm_off_f.close()
    btm_on_f.close()


def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)

