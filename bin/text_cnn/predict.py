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
import text_cnn


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
    # init text cnn model
    model, word_embedding_index, char_embedding_index = text_cnn.init_text_cnn(config)

    # load question ID
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')
    # load dataset
    tc_vecs_on, \
        tw_vecs_on, \
        cc_vecs_on, \
        cw_vecs_on, \
        btm_tw_cw_vecs_on, \
        btm_tc_vecs_on, \
        _ = load_dataset_from_file(config,
                                   'online',
                                   word_embedding_index,
                                   char_embedding_index,
                                   range(len(qid_on)))

    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))

    # load model
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
    model.load(model_fp)
    preds = model.predict([tw_vecs_on, cw_vecs_on, tc_vecs_on, cc_vecs_on, btm_tw_cw_vecs_on, btm_tc_vecs_on],
                          batch_size=batch_size,
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
