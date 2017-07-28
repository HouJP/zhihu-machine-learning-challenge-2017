#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/28 10:42
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys, ConfigParser
from os.path import isfile, join
from keras.models import model_from_json
from os import listdir
import re


import math, heapq
from ..utils import LogUtil, DataUtil
from bin.text_cnn.data_helpers import load_embedding
import data_helpers


def extract_data(regex, content, index=1):
    r = 'nan'
    p = re.compile(regex)
    m = p.search(content)
    if m:
        r = m.group(index)
    return r


def generate_part_ids(config, part_id):
    if -1 != part_id:
        part_ids = [part_id]
    else:
        model_pt = config.get('DIRECTORY', 'model_pt')
        model_files = [f for f in listdir(model_pt) if isfile(join(model_pt, f))]
        part_ids = [extract_data(r'text_cnn_(.*)\.', fn, 1) for fn in model_files]
        part_ids = list(set([int(num) for num in part_ids]))
        part_ids.sort()
    return part_ids


def predict_val(config, part_id):
    version = config.get('TITLE_CONTENT_CNN', 'version')
    data_loader = __import__('bin.text_cnn.%s.data_loader' % version, fromlist=["*"])
    LogUtil.log('INFO', 'version=%s' % version)
    # reset partition size
    config.set('TITLE_CONTENT_CNN', 'part_size', '1024')

    # load word embedding file
    word_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'word_embedding_fn'))
    word_embedding_index, _ = load_embedding(word_embedding_fp)
    # load char embedding file
    char_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'char_embedding_fn'))
    char_embedding_index, _ = load_embedding(char_embedding_fp)

    # init part_ids
    part_ids = generate_part_ids(config, part_id)

    # load offline valid dataset index
    valid_index_off_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                                  config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index_off = DataUtil.load_vector(valid_index_off_fp, 'int')
    valid_index_off = [num - 1 for num in valid_index_off]

    for part_id in part_ids:
        LogUtil.log('INFO', 'part_id=%d' % part_id)

        # load model
        model_fp = config.get('DIRECTORY', 'model_pt') + 'text_cnn_%03d' % part_id
        # load json and create model
        json_file = open('%s.json' % model_fp, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights('%s.h5' % model_fp)
        LogUtil.log('INFO', 'load model (%s) from disk done' % model_fp)

        # make predict and evaluation
        right_label_num = 0
        right_label_at_pos_num = [0] * 5
        sample_num = 0
        all_marked_label_num = 0
        precision = 0.0

        for sub_valid_dataset in data_helpers.load_dataset_from_file_loop(config,
                                                                          'offline',
                                                                          word_embedding_index,
                                                                          char_embedding_index,
                                                                          valid_index_off,
                                                                          False):
            sub_valid_preds = model.predict(sub_valid_dataset[:-1], batch_size=32, verbose=True)

            for i, ps in enumerate(sub_valid_preds):
                sample_num += 1
                top5_ids = [x[0] for x in heapq.nlargest(5, enumerate(ps), key=lambda p: p[1])]

                label_ids = list()
                for kv in enumerate(sub_valid_dataset[-1][i]):
                    if 1 == kv[1]:
                        label_ids.append(kv[0])

                marked_label_set = set(label_ids)
                all_marked_label_num += len(marked_label_set)

                for pos, label in enumerate(top5_ids):
                    if label in marked_label_set:
                        right_label_num += 1
                        right_label_at_pos_num[pos] += 1

        for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
            precision += (right_num / float(sample_num)) / math.log(2.0 + pos)
        recall = float(right_label_num) / all_marked_label_num

        LogUtil.log('INFO', 'precision=%s, recall=%s, f=%s' % (str(precision),
                                                               str(recall),
                                                               str((precision * recall) / (precision + recall))))


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    part_id = int(sys.argv[2])

    predict_val(config, part_id)