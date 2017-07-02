#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/2 16:52
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import json
import sys

from bin.utils import LogUtil
from data_helpers import load_embedding, load_dataset
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
    # load embedding file
    embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'), config.get('TITLE_CONTENT_CNN', 'embedding_fn'))
    embedding_index, embedding_matrix = load_embedding(embedding_fp)
    # load online dataset
    title_length = config.getint('TITLE_CONTENT_CNN', 'title_length')
    content_length = config.getint('TITLE_CONTENT_CNN', 'content_length')
    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')
    que_ids_test, title_vecs_test, cont_vecs_test, label_vecs_test = load_dataset(
        '%s/%s' % (config.get('DIRECTORY', 'dataset_pt'), config.get('TITLE_CONTENT_CNN', 'test_fn')),
        embedding_index,
        class_num,
        title_length,
        content_length)
    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))
    # load model
    optimizer = config.get('TITLE_CONTENT_CNN', 'optimizer')
    metrics = config.get('TITLE_CONTENT_CNN', 'metrics').split()
    batch_size = config.getint('TITLE_CONTENT_CNN', 'batch_size')
    model = TitleContentCNN(title_length=title_length,
                            content_length=content_length,
                            class_num=class_num,
                            embedding_matrix=embedding_matrix,
                            optimizer=optimizer,
                            metrics=metrics)
    model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
    model.load(model_fp)
    preds = model.predict([title_vecs_test, cont_vecs_test], batch_size=batch_size, verbose=True)
    LogUtil.log('INFO', 'prediction of online data, shape=%s' % str(preds.shape))
    # save prediction
    pred_fp = '%s/pred.csv' % config.get('DIRECTORY', 'pred_pt')
    save_prediction(pred_fp, preds, id2label, que_ids_test)


if __name__ == '__main__':
    config_fp = sys.argv[1]
    part_id = int(sys.argv[2])
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    predict(config, part_id)
