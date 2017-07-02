#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/1 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from utils import LogUtil
import numpy as np


def load_embedding(file_path):
    emb_f = open(file_path, 'r')

    shape = emb_f.readline().strip()
    word_num, emb_size = [int(x) for x in shape.split()]
    LogUtil.log('INFO', 'embedding_shape=(%d, %d)' % (word_num, emb_size))

    emb_index = {}
    emb_matrix = [['0.'] * emb_size, ['0.'] * emb_size]

    for line in open(file_path):
        subs = line.strip().split()
        word = subs[0]
        vec = subs[1:]
        emb_index[word] = len(emb_matrix)
        emb_matrix.append(vec)
    emb_matrix = np.asarray(emb_matrix, dtype='float32')

    return emb_index, emb_matrix


def load_valid(valid_fp, emb_index, class_num, size=200):
    qid = []
    title_x_val = []
    cont_x_val = []
    y_val = []

    for line in open(valid_fp):
        line = line.strip('\n')
        part = line.split("\t")
        assert 4 == len(part) == 4

        qid.append(part[0])

        title_word = [emb_index[x] for x in part[1].split(',') if x in emb_index]
        title_word = title_word + [0] * (size - len(title_word)) if len(title_word) < size else title_word[:200]
        cont_word = [emb_index[x] for x in part[2].split(',') if x in emb_index]
        cont_word = cont_word + [0] * (size - len(cont_word)) if len(cont_word) < size else cont_word[:200]
        title_x_val.append(title_word)
        cont_x_val.append(cont_word)

        tmp = [0] * class_num
        for t in part[3].split(','):
            if t == "":
                continue
            tmp[int(t)] = 1
        y_val.append(tmp)

    title_x_val = np.asarray(title_x_val, dtype='int32')
    cont_x_val = np.asarray(cont_x_val, dtype='int32')
    y_val = np.asarray(y_val, dtype='int32')

    LogUtil.log('INFO', 'title_x_val.shape=%s' % str(title_x_val.shape))
    LogUtil.log('INFO', 'cont_x_val.shape=%s' % str(cont_x_val.shape))
    LogUtil.log('INFO', 'y_val.shape=%s' % str(y_val.shape))
    return title_x_val, cont_x_val, y_val, qid


def load_train(file_path, part_size, emb_index, class_num, size=200):
    count = 0
    title_vec = []
    content_vec = []
    label_vec = []
    while True:
        f = open(file_path)
        for line in f:
            line = line.strip('\n')
            part = line.split("\t")
            assert len(part) == 4
            _label_vec = [0] * class_num
            for label_id in part[3].split(','):
                label_id = int(label_id)
                _label_vec[label_id] = 1

            title_word = [emb_index[x] for x in part[1].split(',') if x in emb_index]
            title_word = title_word + [0] * (size - len(title_word)) if len(title_word) < size else title_word[:200]
            cont_word = [emb_index[x] for x in part[2].split(',') if x in emb_index]
            cont_word = cont_word + [0] * (size - len(cont_word)) if len(cont_word) < size else cont_word[:200]

            title_vec.append(title_word)
            content_vec.append(cont_word)
            label_vec.append(_label_vec)

            count += 1
            if 0 == count % part_size:
                title_vec = np.asarray(title_vec, dtype='int32')
                content_vec = np.asarray(content_vec, dtype='int32')
                label_vec = np.asarray(label_vec, dtype='int32')
                yield title_vec, content_vec, label_vec
                title_vec = []
                content_vec = []
                label_vec = []
        f.close()