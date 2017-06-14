#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 23:11
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import data_utils


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
        f.write(' '.join((tw_train_list[i] + dw_train_list[i])) + '\n')
    for i in range(len(qid_eval_list)):
        f.write(' '.join((tw_eval_list[i] + dw_eval_list[i])) + '\n')
    for i in range(len(tid_topic_list)):
        f.write(' '.join((tw_topic_list[i] + dw_topic_list[i])) + '\n')
    f.close()


def main():
    conf_fp = '/home/houjianpeng/zhihu-machine-learning-challenge-2017/conf/default.conf'
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)

    save_question_topic_info(cf)


if __name__ == '__main__':
    main()

