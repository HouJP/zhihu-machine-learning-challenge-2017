#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/2 16:52
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import json
import sys

from data_helpers import *
from ..utils import DataUtil
from bin.evaluation import F


def save_prediction(pred_fp, preds, id2label, que_ids_test):
    pred_f = open(pred_fp, 'w')
    for line_id, p in enumerate(preds):
        label_id_sorted = sorted(list(enumerate(p)), key=lambda s: s[1], reverse=True)
        label_sorted = [id2label[str(kv[0])] for kv in label_id_sorted[:5]]
        pred_f.write("%s,%s\n" % (que_ids_test[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    pred_f.close()

    pred_all_f = open(pred_fp + '.all', 'w')
    for p in preds:
        pred_all_f.write('%s\n', ','.join([str(num) for num in p]))
    pred_all_f.close()


def predict(config, part_id):
    LogUtil.log('INFO', 'part_id=%d' % part_id)

    version = config.get('TITLE_CONTENT_CNN', 'version')
    text_cnn = __import__('bin.text_cnn.%s.text_cnn' % version, fromlist = ["*"])
    data_loader = __import__('bin.text_cnn.%s.data_loader' % version, fromlist = ["*"])

    # init text cnn model
    model, word_embedding_index, char_embedding_index = text_cnn.init_text_cnn(config)

    # load question ID for online dataset
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')

    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
    valid_index_off = [num - 1 for num in valid_index_off]

    # load valid dataset
    valid_dataset = data_loader.load_dataset_from_file(config,
                                                       'offline',
                                                       word_embedding_index,
                                                       char_embedding_index,
                                                       valid_index_off)

    # load test dataset
    test_dataset = data_loader.load_dataset_from_file(config,
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

    # predict for validation
    valid_preds = model.predict(valid_dataset[:-1], batch_size=32, verbose=True)
    LogUtil.log('INFO', 'prediction of validation data, shape=%s' % str(valid_preds.shape))
    F(valid_preds, valid_dataset[-1])
    # predict for test data set
    test_preds = model.predict(test_dataset[:-1], batch_size=32, verbose=True)
    LogUtil.log('INFO', 'prediction of online data, shape=%s' % str(test_preds.shape))
    # save prediction
    pred_fp = '%s/pred.csv.%d' % (config.get('DIRECTORY', 'pred_pt'), part_id)
    save_prediction(pred_fp, test_preds, id2label, qid_on)


if __name__ == '__main__':
    config_fp = sys.argv[1]
    part_id = int(sys.argv[2])
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    predict(config, part_id)
