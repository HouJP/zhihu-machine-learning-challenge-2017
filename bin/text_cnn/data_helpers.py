#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/1 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import numpy as np

from bin.utils import LogUtil


def load_embedding(file_path):
    emb_f = open(file_path, 'r')

    shape = emb_f.readline().strip()
    emb_num, emb_size = [int(x) for x in shape.split()]
    LogUtil.log('INFO', 'embedding_shape=(%d, %d)' % (emb_num, emb_size))

    emb_index = {}
    emb_matrix = [['0.'] * emb_size, ['0.'] * emb_size]

    for line in emb_f:
        subs = line.strip().split()
        word = subs[0]
        vec = subs[1:]
        emb_index[word] = len(emb_matrix)
        emb_matrix.append(vec)
    emb_matrix = np.asarray(emb_matrix, dtype='float32')

    return emb_index, emb_matrix


def parse_dataset_line(line, emb_index, class_num, title_length, content_length, reverse):
    line = line.strip('\n')
    part = line.split("\t")
    assert 4 == len(part)

    que_id = part[0]

    title_vec = [emb_index[x] if x in emb_index else 1 for x in part[1].split(',')]
    if not reverse:
        title_vec = title_vec + [0] * (title_length - len(title_vec)) if len(title_vec) < title_length \
            else title_vec[:title_length]
    else:
        title_vec = [0] * (title_length - len(title_vec)) + title_vec if len(title_vec) < title_length \
            else title_vec[-1 * title_length:]
    cont_vec = [emb_index[x] if x in emb_index else 1 for x in part[2].split(',')]
    if not reverse:
        cont_vec = cont_vec + [0] * (content_length - len(cont_vec)) if len(cont_vec) < content_length \
            else cont_vec[:content_length]
    else:
        cont_vec = [0] * (content_length - len(cont_vec)) + cont_vec if len(cont_vec) < content_length \
            else cont_vec[-1 * content_length:]
    label_vec = [0] * class_num
    if 0 != len(part[3].strip()):
        for label_id in part[3].split(','):
            label_vec[int(label_id)] = 1

    return que_id, title_vec, cont_vec, label_vec


def parse_doc_vec(line, emb_index, vec_length, reverse):
    vec = [emb_index[x] if x in emb_index else 1 for x in line.strip('\n').split(',')]
    if not reverse:
        vec = vec + [0] * (vec_length - len(vec)) if len(vec) < vec_length \
            else vec[:vec_length]
    else:
        vec = [0] * (vec_length - len(vec)) + vec if len(vec) < vec_length \
            else vec[-1 * vec_length:]
    return vec


def load_doc_vec(file_path, emb_index, vec_length, reverse):
    return [parse_doc_vec(line, emb_index, vec_length, reverse) for line in open(file_path).readlines()]


def parse_lid_vec(line, class_num):
    lid_vec = [0] * class_num
    for lid in line.strip('\n').split(','):
        lid_vec[int(lid)] = 1
    return lid_vec


def load_lid(file_path, class_num):
    return [parse_lid_vec(line, class_num) for line in open(file_path).readlines()]


def load_dataset(tc_vecs, tw_vecs, cc_vecs, cw_vecs, btm_vecs, lid_vecs, inds):
    sub_tc_vecs = None if tc_vecs is None else np.asarray([tc_vecs[ind] for ind in inds], dtype='int32')
    sub_tw_vecs = None if tw_vecs is None else np.asarray([tw_vecs[ind] for ind in inds], dtype='int32')
    sub_cc_vecs = None if cc_vecs is None else np.asarray([cc_vecs[ind] for ind in inds], dtype='int32')
    sub_cw_vecs = None if cw_vecs is None else np.asarray([cw_vecs[ind] for ind in inds], dtype='int32')
    sub_btm_vecs = None if cw_vecs is None else np.asarray([btm_vecs[ind] for ind in inds], dtype='int32')
    sub_lid_vecs = None if lid_vecs is None else np.asarray([lid_vecs[ind] for ind in inds], dtype='int32')
    return sub_tc_vecs, sub_tw_vecs, sub_cc_vecs, sub_cw_vecs, sub_btm_vecs, sub_lid_vecs


def load_dataset_loop(tc_vecs, tw_vecs, cc_vecs, cw_vecs, btm_vecs, lid_vecs, inds, part_size):
    count = 0
    inds_len = len(inds)
    inds_part = list()
    while True:
        count += 1
        inds_part.append(inds[count % inds_len])
        if 0 == count % part_size:
            yield load_dataset(tc_vecs, tw_vecs, cc_vecs, cw_vecs, btm_vecs, lid_vecs, inds_part)
            inds_part = list()


if __name__ == '__main__':
    load_embedding('/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/data/embedding/word_embedding.txt.small')

