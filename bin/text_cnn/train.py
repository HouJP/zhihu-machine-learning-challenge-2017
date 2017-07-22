#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 16:41
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import os
import sys
import time

from ..utils import DataUtil
from data_helpers import *
import text_cnn


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
    # init text cnn model
    model, word_embedding_index, char_embedding_index = text_cnn.init_text_cnn(config)
    # init directory
    init_out_dir(config)

    # load offline train dataset index
    train_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'train_index_offline_fn'))
    train_index_off = DataUtil.load_vector(train_index_off_fp, 'int')
    train_index_off = [num - 1 for num in train_index_off]

    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
    valid_index_off = [num - 1 for num in valid_index_off]

    # load valid dataset
    valid_tc_vecs, \
        valid_tw_vecs, \
        valid_cc_vecs, \
        valid_cw_vecs, \
        valid_btm_tw_cw, \
        valid_lid_vecs = load_dataset_from_file(config,
                                                'offline',
                                                word_embedding_index,
                                                char_embedding_index,
                                                valid_index_off)

    # load train dataset
    part_id = 0
    part_size = config.getint('TITLE_CONTENT_CNN', 'part_size')
    valid_size = config.getint('TITLE_CONTENT_CNN', 'valid_size')
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    for train_tc_vecs, \
        train_tw_vecs, \
        train_cc_vecs, \
        train_cw_vecs, \
        train_btm_tw_cw, \
        train_lid_vecs in load_dataset_from_file_loop(config,
                                                      'offline',
                                                      word_embedding_index,
                                                      char_embedding_index,
                                                      train_index_off):
        LogUtil.log('INFO', 'part_id=%d, model training begin' % part_id)
        if 0 == (((part_id + 1) * part_size) % valid_size):
            model.fit([train_tw_vecs, train_cw_vecs, train_tc_vecs, train_cc_vecs, train_btm_tw_cw],
                      train_lid_vecs,
                      validation_data=(
                          [valid_tw_vecs, valid_cw_vecs, valid_tc_vecs, valid_cc_vecs, valid_btm_tw_cw],
                          valid_lid_vecs),
                      epochs=1,
                      batch_size=batch_size)
            model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
            model.save(model_fp)
        else:
            model.fit(
                [train_tw_vecs, train_cw_vecs, train_tc_vecs, train_cc_vecs, train_btm_tw_cw],
                train_lid_vecs,
                epochs=1,
                batch_size=batch_size)
        part_id += 1


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    train(config)
