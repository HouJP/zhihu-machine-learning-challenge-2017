#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 7/25/17 9:15 PM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com


from bin.text_cnn.data_helpers import *


def load_dataset_from_file(config, data_name, word_emb_index, char_emb_index, inds):
    # make a copy of index
    inds_sorted = sorted(enumerate(inds), key=lambda kv: kv[1])
    inds_copy = [kv[1] for kv in inds_sorted]
    inds_map = [kv[0] for kv in inds_sorted]

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
    # load btm_tc features
    fs_btm_tc_fp = '%s/%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'fs_btm_tc', data_name)
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

    sub_fs_btm_tc = np.asarray(load_feature_vec_part(fs_btm_tc_fp, inds_copy, inds_map), dtype='float32')
    LogUtil.log('INFO', 'load btm_tc features done')

    sub_fs_word_share = np.asarray(load_feature_vec_part(fs_word_share_fp, inds_copy, inds_map), dtype='float32')
    LogUtil.log('INFO', 'load word_share features done')

    sub_lid_vecs = None if lid_fp is None else np.asarray(load_lid_part(lid_fp, class_num, inds_copy, inds_map), dtype='int32')
    LogUtil.log('INFO', 'load label id vector done')

    return [sub_tw_vecs, sub_cw_vecs, sub_tc_vecs, sub_cc_vecs, sub_fs_btm_tw_cw, sub_fs_btm_tc, sub_fs_word_share, sub_lid_vecs]

