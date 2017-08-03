#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/1 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import numpy as np
import math
import random
import re
from bin.utils import LogUtil
from os.path import isfile


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

def load_embedding_with_idx(file_path, emb_index):
    emb_f = open(file_path, 'r')

    shape = emb_f.readline().strip()
    emb_num, emb_size = [int(x) for x in shape.split()]
    LogUtil.log('INFO', 'embedding_shape=(%d, %d)' % (emb_num, emb_size))

    emb_matrix = np.zeros([emb_num+2, emb_size])

    for line in emb_f:
        subs = line.strip().split()
        word = subs[0]
        vec = subs[1:]
        if word in emb_index:
            emb_matrix[emb_index[word]] = np.asarray(vec)

    return emb_matrix

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
    vec = [0. if math.isnan(float(num)) else float(num) for num in re.split(' |,', line.strip())]
    return vec


def parse_feature_sparse_vec(line, length):
    vec = [0.] * length
    if 0 == len(line.strip()):
        return vec
    else:
        for kv in re.split(' |,', line.strip()):
            fid, fv = kv.split(':')
            vec[int(fid)] = 0. if math.isnan(float(fv)) else float(fv)
        return vec


def load_feature_vec(file_path):
    if isfile(file_path + '.smat'):
        LogUtil.log('INFO', 'load sparse feature file %s' % file_path)
        f = open(file_path + '.smat', 'r')
        row_num, col_num = re.split(' |,', f.readline().strip('\n'))
        return [parse_feature_sparse_vec(line, int(col_num)) for line in f.readlines()]
    else:
        LogUtil.log('INFO', 'load dense feature file %s' % file_path)
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
            doc_vecs[inds_map[index_inds]] = parse_doc_vec(line, emb_index, vec_length, reverse)
            index_inds += 1
        index_f += 1
    f.close()

    return doc_vecs


def load_feature_vec_part(file_path, inds_copy, inds_map):
    vecs = [0] * len(inds_copy)

    index_f = 0
    index_inds = 0

    is_smat = isfile('%s.smat' % file_path)

    if is_smat:
        LogUtil.log('INFO', 'load sparse feature file %s' % file_path)
        f = open('%s.smat' % file_path, 'r')
        row_num, col_num = re.split(' |,', f.readline().strip('\n'))
        row_num = int(row_num)
        col_num = int(col_num)
    else:
        LogUtil.log('INFO', 'load dense feature file %s' % file_path)
        f = open(file_path, 'r')
        row_num = col_num = -1

    for line in f:
        if len(inds_copy) <= index_inds:
            break
        if index_f == inds_copy[index_inds]:
            vecs[inds_map[index_inds]] = parse_feature_vec(line) if not is_smat else parse_feature_sparse_vec(line, col_num)
            index_inds += 1
        index_f += 1
    f.close()

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
            vecs[inds_map[index_inds]] = parse_lid_vec(line, class_num)
            index_inds += 1
        index_f += 1
    f.close()

    return vecs


def load_raw_line_part(file_path, inds_copy, inds_map):
    vecs = [0] * len(inds_copy)

    index_f = 0
    index_inds = 0
    f = open(file_path, 'r')
    for line in f:
        if len(inds_copy) <= index_inds:
            break
        if index_f == inds_copy[index_inds]:
            vecs[inds_map[index_inds]] = line
            index_inds += 1
        index_f += 1
    f.close()

    return vecs


def load_features_from_file(config, feature_name, data_name, inds):
    # make a copy of index
    inds_sorted = sorted(enumerate(inds), key=lambda kv: kv[1])
    inds_copy = [kv[1] for kv in inds_sorted]
    inds_map = [kv[0] for kv in inds_sorted]

    # load features
    feature_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, data_name)

    sub_features = load_feature_vec_part(feature_fp, inds_copy, inds_map)
    LogUtil.log('INFO', 'len(sub_features)=%d' % len(sub_features))
    sub_features = np.asarray(sub_features, dtype='float32')
    LogUtil.log('INFO', 'load features done')

    return sub_features


def load_labels_from_file(config, data_name, inds):
    # make a copy of index
    inds_sorted = sorted(enumerate(inds), key=lambda kv: kv[1])
    inds_copy = [kv[1] for kv in inds_sorted]
    inds_map = [kv[0] for kv in inds_sorted]

    # load label id vectors
    lid_fp = None if 'online' == data_name \
        else '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'label_id', data_name)

    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')

    sub_lid_vecs = None if lid_fp is None else np.asarray(load_lid_part(lid_fp, class_num, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load label id vector done')

    return sub_lid_vecs


def load_raw_line_from_file(config, file_path, inds):
    # make a copy of index
    inds_sorted = sorted(enumerate(inds), key=lambda kv: kv[1])
    inds_copy = [kv[1] for kv in inds_sorted]
    inds_map = [kv[0] for kv in inds_sorted]

    sub_lines = load_raw_line_part(file_path, inds_copy, inds_map)

    LogUtil.log('INFO', 'load raw line done')

    return sub_lines


def load_dataset_from_file_loop(config, data_name, word_emb_index, char_emb_index, inds, loop=True):
    version = config.get('TITLE_CONTENT_CNN', 'version')
    LogUtil.log('INFO', 'version=%s' % version)
    data_loader = __import__('bin.text_cnn.%s.data_loader' % version, fromlist=["*"])
    part_size = config.getint('TITLE_CONTENT_CNN', 'part_size')

    inds_len = len(inds)
    inds_index = 0

    sub_inds = list()

    while True:

        if inds_len <= inds_index:
            if loop:
                inds_index = 0
                random.shuffle(inds)
            else:
                break

        sub_inds.append(inds[inds_index])
        inds_index += 1

        if (part_size == len(sub_inds)) or (inds_len <= inds_index):
            # delete duplicate
            sub_inds = reduce(lambda x, y: x if y in x else x + [y], [[], ] + sub_inds)
            yield data_loader.load_dataset_from_file(config, data_name, word_emb_index, char_emb_index, sub_inds)
            sub_inds = list()


if __name__ == '__main__':
    load_embedding(
        '/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/data/embedding/word_embedding.txt.small')
