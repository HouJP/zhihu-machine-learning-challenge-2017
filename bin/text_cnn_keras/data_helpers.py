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

    for line in emb_f:
        subs = line.strip().split()
        word = subs[0]
        vec = subs[1:]
        emb_index[word] = len(emb_matrix)
        emb_matrix.append(vec)
    emb_matrix = np.asarray(emb_matrix, dtype='float32')

    return emb_index, emb_matrix


def parse_dataset_line(line, emb_index, class_num, title_length, content_length):
    line = line.strip('\n')
    part = line.split("\t")
    assert 4 == len(part)

    que_id = part[0]

    title_vec = [emb_index[x] if x in emb_index else 1 for x in part[1].split(',')]
    title_vec = title_vec + [0] * (title_length - len(title_vec)) if len(title_vec) < title_length \
        else title_vec[:title_length]
    cont_vec = [emb_index[x] if x in emb_index else 1 for x in part[2].split(',')]
    cont_vec = cont_vec + [0] * (content_length - len(cont_vec)) if len(cont_vec) < content_length \
        else cont_vec[:content_length]

    label_vec = [0] * class_num
    for label_id in part[3].split(','):
        label_vec[int(label_id)] = 1

    return que_id, title_vec, cont_vec, label_vec


def load_dataset(file_path, emb_index, class_num, title_length, content_length):
    que_ids = []
    title_vecs = []
    cont_vecs = []
    label_vecs = []

    for line in open(file_path):
        que_id, title_vec, cont_vec, label_vec = parse_dataset_line(line,
                                                                    emb_index,
                                                                    class_num,
                                                                    title_length,
                                                                    content_length)

        que_ids.append(que_id)
        title_vecs.append(title_vec)
        cont_vecs.append(cont_vec)
        label_vecs.append(label_vec)

    title_vecs = np.asarray(title_vecs, dtype='int32')
    cont_vecs = np.asarray(cont_vecs, dtype='int32')
    label_vecs = np.asarray(label_vecs, dtype='int32')

    LogUtil.log('INFO', 'title_vecs.shape=%s' % str(title_vecs.shape))
    LogUtil.log('INFO', 'cont_vecs.shape=%s' % str(cont_vecs.shape))
    LogUtil.log('INFO', 'label_vecs.shape=%s' % str(label_vecs.shape))
    return title_vecs, cont_vecs, label_vecs, que_ids


def load_dataset_loop(file_path, part_size, emb_index, class_num, title_length, content_length):
    count = 0
    que_ids = []
    title_vecs = []
    cont_vecs = []
    label_vecs = []
    while True:
        f = open(file_path)
        for line in f:
            que_id, title_vec, cont_vec, label_vec = parse_dataset_line(line,
                                                                        emb_index,
                                                                        class_num,
                                                                        title_length,
                                                                        content_length)

            que_ids.append(que_id)
            title_vecs.append(title_vec)
            cont_vecs.append(cont_vec)
            label_vecs.append(label_vec)

            count += 1
            if 0 == count % part_size:
                title_vecs = np.asarray(title_vecs, dtype='int32')
                cont_vecs = np.asarray(cont_vecs, dtype='int32')
                label_vecs = np.asarray(label_vecs, dtype='int32')
                yield title_vecs, cont_vecs, label_vecs, que_ids
                title_vecs = []
                cont_vecs = []
                label_vecs = []
                que_ids = []
        f.close()


if __name__ == '__main__':
    load_embedding('/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/data/embedding/word_embedding.txt.small')

