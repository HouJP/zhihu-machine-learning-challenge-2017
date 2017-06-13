#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 21:39
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

import ConfigParser


def load_question_set(fp):
    """
    load `question_train_set.txt` and `question_eval_set.txt`
    :param fp:
    :return:
    """
    f = open(fp)
    qid_list = []
    tc_list = []
    tw_list = []
    dc_list = []
    dw_list = []
    for line in f:
        qid, c = line.strip().split('\t', 1)
        qid_list.append(qid)
        subs_c = c.split('\t')
        if 0 < len(subs_c):
            tc_list.append(subs_c[0].split(','))
        else:
            tc_list.append([])
        if 1 < len(subs_c):
            tw_list.append(subs_c[1].split(','))
        else:
            tw_list.append([])
        if 2 < len(subs_c):
            dc_list.append(subs_c[2].split(','))
        else:
            dc_list.append([])
        if 3 < len(subs_c):
            dw_list.append(subs_c[3].split(','))
        else:
            dw_list.append([])
    f.close()
    return qid_list, tc_list, tw_list, dc_list, dw_list


def _test_load_question_set():
    q_train_set = cf.get('DEFAULT', 'source_pt') + '/question_train_set.txt.small'

    (qid_list, tc_list, tw_list, dc_list, dw_list) = load_question_set(q_train_set)
    print qid_list
    print tc_list
    print tw_list
    print dc_list
    print dw_list

if __name__ == '__main__':
    conf_fp = '/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/conf/default.conf'
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)

    _test_load_question_set()
