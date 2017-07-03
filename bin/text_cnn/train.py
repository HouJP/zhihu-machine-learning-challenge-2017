#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 16:41
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import os
import sys
import time

from ..utils import LogUtil
from data_helpers import load_dataset, load_dataset_loop
from data_helpers import load_embedding
from text_cnn import TitleContentCNN


def init_out_dir(config):
    # generate output tag
    out_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    config.set('DIRECTORY', 'out_tag', str(out_tag))
    # generate output directory
    out_pt = config.get('DIRECTORY', 'out_pt')
    out_pt_exists = os.path.exists(out_pt)
    if out_pt_exists:
        LogUtil.log("ERROR", 'out path (%s) already exists ' % out_pt)
        raise Exception
    else:
        os.mkdir(out_pt)
        os.mkdir(config.get('DIRECTORY', 'pred_pt'))
        os.mkdir(config.get('DIRECTORY', 'model_pt'))
        os.mkdir(config.get('DIRECTORY', 'conf_pt'))
        os.mkdir(config.get('DIRECTORY', 'score_pt'))
        LogUtil.log('INFO', 'out path (%s) created ' % out_pt)
    # save config
    config.write(open(config.get('DIRECTORY', 'conf_pt') + 'featwheel.conf', 'w'))


def train(config):
    init_out_dir(config)
    # load embedding file
    embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'), config.get('TITLE_CONTENT_CNN', 'embedding_fn'))
    embedding_index, embedding_matrix = load_embedding(embedding_fp)
    # init model
    title_length = config.getint('TITLE_CONTENT_CNN', 'title_length')
    content_length = config.getint('TITLE_CONTENT_CNN', 'content_length')
    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')
    optimizer = config.get('TITLE_CONTENT_CNN', 'optimizer')
    metrics = config.get('TITLE_CONTENT_CNN', 'metrics').split()
    model = TitleContentCNN(title_length=title_length,
                            content_length=content_length,
                            class_num=class_num,
                            embedding_matrix=embedding_matrix,
                            optimizer=optimizer,
                            metrics=metrics)

    que_ids_valid, title_vecs_valid, cont_vecs_valid, label_vecs_valid = load_dataset(
        '%s/%s' % (config.get('DIRECTORY', 'dataset_pt'), config.get('TITLE_CONTENT_CNN', 'valid_fn')),
        embedding_index,
        class_num,
        title_length,
        content_length)

    part_id = 0
    part_size = config.getint('TITLE_CONTENT_CNN', 'part_size')
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    train_fp = '%s/%s' % (config.get('DIRECTORY', 'dataset_pt'), config.get('TITLE_CONTENT_CNN', 'train_fn'))
    for que_ids_train, title_vecs_train, cont_vecs_train, label_vecs_train in load_dataset_loop(train_fp,
                                                                                                part_size,
                                                                                                embedding_index,
                                                                                                class_num,
                                                                                                title_length,
                                                                                                content_length):
        LogUtil.log('INFO', 'part_id=%d, model training begin' % part_id)
        model.fit([title_vecs_train, cont_vecs_train],
                  label_vecs_train,
                  validation_data=([title_vecs_valid, cont_vecs_valid], label_vecs_valid),
                  epochs=1,
                  batch_size=batch_size)
        model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
        model.save(model_fp)
        part_id += 1


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    train(config)
