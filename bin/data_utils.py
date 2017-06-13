#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 21:39
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


def load_question_set(fp):
    f = open(fp)
    qid_list = []
    cs_list = []
    for line in f:
        qid, c = line.strip().split('\t', 1)
        cs = c.split('\t')
        for i in range(len(cs)):
            cs[i] = cs[i].split(',')
        qid_list.append(qid)
        cs_list.append(cs)
    f.close()
    return qid_list, cs_list

