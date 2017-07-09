#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/2 16:52
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import json
import sys

from ..utils import LogUtil, DataUtil
from data_helpers import *
from text_cnn import TitleContentCNN


def save_prediction(pred_fp, preds, id2label, que_ids_test):
    pred_f = open(pred_fp, 'w')
    for line_id, p in enumerate(preds):
        label_id_sorted = sorted(list(enumerate(p)), key=lambda s: s[1], reverse=True)
        label_sorted = [id2label[str(kv[0])] for kv in label_id_sorted[:5]]
        pred_f.write("%s,%s\n" % (que_ids_test[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    pred_f.close()


def predict(config, part_id):
    # load word embedding file
    word_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'word_embedding_fn'))
    word_embedding_index, word_embedding_matrix = load_embedding(word_embedding_fp)
    # load char embedding file
    char_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'char_embedding_fn'))
    char_embedding_index, char_embedding_matrix = load_embedding(char_embedding_fp)
    # load btm embedding file
    btm_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                  config.get('TITLE_CONTENT_CNN', 'btm_embedding_fn'))
    btm_embedding_index, btm_embedding_matrix = load_embedding(btm_embedding_fp)
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
                            btm_embedding_matrix=btm_embedding_matrix,
                            optimizer=optimizer,
                            metrics=metrics)
    # load title char vectors
    tc_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_char')
    tc_vecs_on = load_doc_vec(tc_on_fp, char_embedding_index, title_char_length, reverse=True)
    LogUtil.log('INFO', 'load online title char vector done')

    # load title word vectors
    tw_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'title_word')
    tw_vecs_on = load_doc_vec(tw_on_fp, word_embedding_index, title_word_length, reverse=False)
    LogUtil.log('INFO', 'load online title word vector done')

    # load content char vectors
    cc_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_char')
    cc_vecs_on = load_doc_vec(cc_on_fp, char_embedding_index, content_char_length, reverse=True)
    LogUtil.log('INFO', 'load online content char vector done')

    # load content word vectors
    cw_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'content_word')
    cw_vecs_on = load_doc_vec(cw_on_fp, word_embedding_index, content_word_length, reverse=False)
    LogUtil.log('INFO', 'load online content word vector done')

    # load btm vectors
    btm_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'btm')
    btm_vecs_on = load_doc_vec(btm_on_fp, btm_embedding_index, 1, reverse=False)
    LogUtil.log('INFO', 'load online btm vector done')

    # load question ID
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')

    tc_vecs_on, tw_vecs_on, cc_vecs_on, cw_vecs_on, btm_vecs_on, _ = load_dataset(tc_vecs_on, tw_vecs_on, cc_vecs_on,
                                                                                  cw_vecs_on,
                                                                                  btm_vecs_on,
                                                                                  None,
                                                                                  range(len(qid_on)))

    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))

    # load model
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
    model.load(model_fp)
    preds = model.predict([tw_vecs_on, cw_vecs_on, tc_vecs_on, cc_vecs_on, btm_vecs_on], batch_size=batch_size,
                          verbose=True)
    LogUtil.log('INFO', 'prediction of online data, shape=%s' % str(preds.shape))
    # save prediction
    pred_fp = '%s/pred.csv' % config.get('DIRECTORY', 'pred_pt')
    save_prediction(pred_fp, preds, id2label, qid_on)


if __name__ == '__main__':
    config_fp = sys.argv[1]
    part_id = int(sys.argv[2])
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    predict(config, part_id)
