#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/1 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from utils import LogUtil
import numpy as np


def load_embedding_file(file_path):
    emb_size = 0
    word_num = 0

    for line in open(file_path):
        values = line.split()
        if len(values) == 0:
            continue
        word_num += 1
        if 0 == emb_size:
            coefs = np.asarray(values[1:], dtype='float32')
            emb_size = len(coefs)

    LogUtil.log('INFO', 'word_num=%d' % word_num)
    LogUtil.log('INFO', 'emb_size=%d' % emb_size)

    embedding_index = {}
    embedding_matrix = np.zeros((word_num + 1, emb_size))

    cnt = 0
    for line in open(file_path):
        values = line.split()
        if len(values) == 0:
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        cnt += 1
        embedding_index[word] = cnt
        embedding_matrix[cnt] = coefs

    LogUtil.log('INFO', "Total embedding word is %s ." % len(embedding_index))
    return embedding_index, embedding_matrix


def load_valid(valid_fp, emb_index, size=200):
    qid = []
    title_x_val = []
    cont_x_val = []
    y_val = []

    for line in open(valid_fp):
        line = line.strip('\n')
        part = line.split("\t")
        assert (len(part) == 4)

        qid.append(part[0])

        title_word = [emb_index[x] for x in part[1].split(',') if x in emb_index]
        title_word = title_word + [0] * (size - len(title_word)) if len(title_word) < size else title_word[:200]
        cont_word = [emb_index[x] for x in part[2].split(',') if x in emb_index]
        cont_word = cont_word + [0] * (size - len(cont_word)) if len(cont_word) < size else cont_word[:200]
        title_x_val.append(title_word)
        cont_x_val.append(cont_word)

        tmp = [0] * 2000
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


def load_train(file_path, part_size, emb_index, size=200):
    count = 0
    title_vec = []
    content_vec = []
    label_vec = []
    while True:
        f = open(file_path)
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            part = line.split("\t")
            _label_vec = [0] * 2000
            for label_id in part[3].split(','):
                label_id = int(label_id)
                assert label_id != 0, 'illegal label ID (0)'
                _label_vec[label_id] = 1

            title_word = [emb_index[x] for x in part[1].split(',') if x in emb_index]
            title_word = title_word + [0] * (size - len(title_word)) if len(title_word) < size else title_word[:200]
            cont_word = [emb_index[x] for x in part[2].split(',') if x in emb_index]
            cont_word = cont_word + [0] * (size - len(cont_word)) if len(cont_word) < size else cont_word[:200]

            title_vec.append(title_word)
            content_vec.append(cont_word)
            label_vec.append(_label_vec)

            count += 1
            if count == part_size:
                title_vec = np.asarray(title_vec, dtype='int32')
                content_vec = np.asarray(content_vec, dtype='int32')
                label_vec = np.asarray(label_vec, dtype='int32')
                yield title_vec, content_vec, label_vec
                count = 0
                title_vec = []
                content_vec = []
                label_vec = []
        f.close()