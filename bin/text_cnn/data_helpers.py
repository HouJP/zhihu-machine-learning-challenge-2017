#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/1 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import numpy as np
import math
import random
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


def parse_feature_vec(line):
    vec = [0. if math.isnan(float(num)) else float(num) for num in line.strip('\n').split()]
    return vec


def load_feature_vec(file_path):
    return [parse_feature_vec(line) for line in open(file_path).readlines()]


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
    sub_btm_vecs = None if btm_vecs is None else np.asarray([btm_vecs[ind] for ind in inds], dtype='float32')
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


def load_doc_vec_part(file_path, emb_index, vec_length, reverse, inds_copy, inds_map):
    doc_vecs = [0] * len(inds_copy)

    index_f = 0
    index_inds = 0
    f = open(file_path, 'r')
    for line in f:
        if len(inds_copy) <= index_inds:
            break
        if index_f == inds_copy[index_inds]:
            doc_vecs[index_inds] = parse_doc_vec(line, emb_index, vec_length, reverse)
            index_inds += 1
        index_f += 1
    f.close()

    doc_vecs = [doc_vecs[i] for i in inds_map]

    return doc_vecs


def load_feature_vec_part(file_path, inds_copy, inds_map):
    vecs = [0] * len(inds_copy)

    index_f = 0
    index_inds = 0
    f = open(file_path, 'r')
    for line in f:
        if len(inds_copy) <= index_inds:
            break
        if index_f == inds_copy[index_inds]:
            vecs[index_inds] = parse_feature_vec(line)
            index_inds += 1
        index_f += 1
    f.close()

    vecs = [vecs[i] for i in inds_map]

    return vecs


def load_lid_part(file_path, class_num, inds_copy, inds_map):
    vecs = [0] * len(inds_copy)

    index_f = 0
    index_inds = 0
    f = open(file_path, 'r')
    for line in f:
        if len(inds_copy) <= index_inds:
            break
        if index_f == inds_copy[index_inds]:
            vecs[index_inds] = parse_lid_vec(line, class_num)
            index_inds += 1
        index_f += 1
    f.close()

    vecs = [vecs[i] for i in inds_map]

    return vecs


def load_dataset_from_file(config, data_name, word_emb_index, char_emb_index, inds):
    # make a copy of index
    inds_sorted = sorted(enumerate(inds), key=lambda kv: kv[1])
    inds_copy = [kv[1] for kv in inds_sorted]
    inds_map = [kv2[0] for kv2 in sorted(enumerate([kv3[0] for kv3 in inds_sorted]), key=lambda kv: kv[1])]

    # load title char vectors
    tc_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_char', data_name)
    # load title word vectors
    tw_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_word', data_name)
    # load content char vectors
    cc_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_char', data_name)
    # load content word vectors
    cw_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_word', data_name)
    # load btm_tw_cw features
    fs_btm_tw_cw_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'fs_btm_tw_cw', data_name)
    # load word_share features
    fs_word_share_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'fs_word_share', data_name)
    # load label id vectors
    lid_fp = None if 'online' == data_name \
        else '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'label_id', data_name)

    title_word_length = config.getint('TITLE_CONTENT_CNN', 'title_word_length')
    content_word_length = config.getint('TITLE_CONTENT_CNN', 'content_word_length')
    title_char_length = config.getint('TITLE_CONTENT_CNN', 'title_char_length')
    content_char_length = config.getint('TITLE_CONTENT_CNN', 'content_char_length')
    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')

    sub_tc_vecs = np.asarray(load_doc_vec_part(tc_fp, char_emb_index, title_char_length, True, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load title char vector done')

    sub_tw_vecs = np.asarray(load_doc_vec_part(tw_fp, word_emb_index, title_word_length, False, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load title word vector done')

    sub_cc_vecs = np.asarray(load_doc_vec_part(cc_fp, char_emb_index, content_char_length, True, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load content char vector done')

    sub_cw_vecs = np.asarray(load_doc_vec_part(cw_fp, word_emb_index, content_word_length, False, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load content word vector done')

    sub_fs_btm_tw_cw = np.asarray(load_feature_vec_part(fs_btm_tw_cw_fp, inds_copy, inds_map), dtype='float32')
    LogUtil.log('INFO', 'load btm_tw_cw features done')

    sub_fs_word_share = np.asarray(load_feature_vec_part(fs_word_share_fp, inds_copy, inds_map), dtype='float32')
    LogUtil.log('INFO', 'load word share features done')

    sub_lid_vecs = None if lid_fp is None else np.asarray(load_lid_part(lid_fp, class_num, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load label id vector done')

    return sub_tc_vecs, sub_tw_vecs, sub_cc_vecs, sub_cw_vecs, sub_fs_btm_tw_cw, sub_fs_word_share, sub_lid_vecs


def load_dataset_from_file_loop(config, data_name, word_emb_index, char_emb_index, inds):
    part_size = config.getint('TITLE_CONTENT_CNN', 'part_size')

    inds_len = len(inds)
    inds_index = 0

    sub_inds = list()

    while True:

        if inds_len <= inds_index:
            inds_index = 0
            random.shuffle(inds)

        sub_inds.append(inds[inds_index])
        inds_index += 1

        if part_size == len(sub_inds):
            yield load_dataset_from_file(config, data_name, word_emb_index, char_emb_index, sub_inds)
            sub_inds = list()


if __name__ == '__main__':
    load_embedding('/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/data/embedding/word_embedding.txt.small')

