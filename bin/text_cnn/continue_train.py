#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/15 21:06
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import os
import sys
import time

from ..utils import DataUtil
from data_helpers import *
import text_cnn


def train(config, part_id):
    # init text cnn model
    model, word_embedding_index, char_embedding_index = text_cnn.init_text_cnn(config)

    # load model
    model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
    model.load(model_fp)
    part_id += 1

    # load offline train dataset index
    train_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'train_index_offline_fn'))
    train_index_off = DataUtil.load_vector(train_index_off_fp, 'int')

    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')

    # load valid dataset
    valid_tc_vecs, \
        valid_tw_vecs, \
        valid_cc_vecs, \
        valid_cw_vecs, \
        valid_btm_tw_cw_vecs, \
        valid_btm_tc_vecs, \
        valid_lid_vecs = load_dataset_from_file(config,
                                                'offline',
                                                word_embedding_index,
                                                char_embedding_index,
                                                valid_index_off)

    # load train dataset
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    for train_tc_vecs, \
        train_tw_vecs, \
        train_cc_vecs, \
        train_cw_vecs, \
        train_btm_tw_cw_vecs,\
        train_btm_tc_vecs, \
        train_lid_vecs in load_dataset_from_file_loop(config,
                                                      'offline',
                                                      word_embedding_index,
                                                      char_embedding_index,
                                                      train_index_off,
                                                      part_id):
        LogUtil.log('INFO', 'part_id=%d, model training begin' % part_id)
        model.fit([train_tw_vecs, train_cw_vecs, train_tc_vecs, train_cc_vecs, train_btm_tw_cw_vecs, train_btm_tc_vecs],
                  train_lid_vecs,
                  validation_data=(
                      [valid_tw_vecs, valid_cw_vecs, valid_tc_vecs, valid_cc_vecs, valid_btm_tw_cw_vecs,
                       valid_btm_tc_vecs], valid_lid_vecs),
                  epochs=1,
                  batch_size=batch_size)
        model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
        model.save(model_fp)
        part_id += 1


if __name__ == '__main__':
    config_fp = sys.argv[1]
    part_id = int(sys.argv[2])
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    predict(config, part_id)