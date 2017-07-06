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
    # load word embedding file
    word_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'word_embedding_fn'))
    word_embedding_index, word_embedding_matrix = load_embedding(word_embedding_fp)
    # load char embedding file
    char_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'char_embedding_fn'))
    char_embedding_index, char_embedding_matrix = load_embedding(char_embedding_fp)
    # init model
    title_word_length = config.getint('TITLE_CONTENT_CNN', 'title_word_length')
    content_word_length = config.getint('TITLE_CONTENT_CNN', 'content_word_length')
    title_char_length = config.getint('TITLE_CONTENT_CNN', 'title_char_length')
    content_char_length = config.getint('TITLE_CONTENT_CNN', 'content_char_length')
    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')
    optimizer = config.get('TITLE_CONTENT_CNN', 'optimizer')
    metrics = config.get('TITLE_CONTENT_CNN', 'metrics').split()
    model = TitleContentCNN(title_word_length=title_word_length,
                            content_word_length=content_word_length,
                            title_char_length=title_char_length,
                            content_char_length=content_char_length,
                            class_num=class_num,
                            word_embedding_matrix=word_embedding_matrix,
                            char_embedding_matrix=char_embedding_matrix,
                            optimizer=optimizer,
                            metrics=metrics)

    # load title char vectors
    tc_off_fp = '%s/%s.offline.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_char')
    tc_vecs_off = load_doc_vec(tc_off_fp, char_embedding_matrix, title_char_length, reverse=True)
    LogUtil.log('INFO', 'load offline title char vector done')

    # load title word vectors
    tw_off_fp = '%s/%s.offline.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_word')
    tw_vecs_off = load_doc_vec(tw_off_fp, word_embedding_matrix, title_word_length, reverse=False)
    LogUtil.log('INFO', 'load offline title word vector done')

    # load content char vectors
    cc_off_fp = '%s/%s.offline.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_char')
    cc_vecs_off = load_doc_vec(cc_off_fp, char_embedding_matrix, content_char_length, reverse=True)
    LogUtil.log('INFO', 'load offline content char vector done')

    # load content word vectors
    cw_off_fp = '%s/%s.offline.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_word')
    cw_vecs_off = load_doc_vec(cw_off_fp, word_embedding_matrix, content_word_length, reverse=False)
    LogUtil.log('INFO', 'load offline content word vector done')

    # load label id vectors
    lid_off_fp = '%s/%s.offline.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'label_id')
    lid_vecs_off = load_lid(lid_off_fp, class_num)
    LogUtil.log('INFO', 'load offline label id vector done')

    # load offline train dataset index
    train_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'train_index_offline_fn'))
    train_index_off = DataUtil.load_vector(train_index_off_fp, 'int')

    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')

    # load valid dataset
    valid_tc_vecs, valid_tw_vecs, valid_cc_vecs, valid_cw_vecs, valid_lid_vecs = load_dataset(tc_vecs_off,
                                                                                              tw_vecs_off,
                                                                                              cc_vecs_off,
                                                                                              cw_vecs_off,
                                                                                              lid_vecs_off,
                                                                                              valid_index_off)

    # load train dataset
    part_id = 0
    part_size = config.getint('TITLE_CONTENT_CNN', 'part_size')
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    for train_tc_vecs, train_tw_vecs, train_cc_vecs, train_cw_vecs, train_lid_vecs in load_dataset_loop(tc_vecs_off,
                                                                                                        tw_vecs_off,
                                                                                                        cc_vecs_off,
                                                                                                        cw_vecs_off,
                                                                                                        lid_vecs_off,
                                                                                                        train_index_off,
                                                                                                        part_size):
        LogUtil.log('INFO', 'part_id=%d, model training begin' % part_id)
        model.fit([train_tw_vecs, train_cw_vecs, train_tc_vecs, train_cc_vecs],
                  train_lid_vecs,
                  validation_data=([valid_tw_vecs, valid_cw_vecs, valid_tc_vecs, valid_cc_vecs], valid_lid_vecs),
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
