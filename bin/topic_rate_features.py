#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/14 22:18
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from data_utils import load_question_topic_set
import json
import ConfigParser
import sys
from featwheel.feature import Feature


def generate(config, argv):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/question_topic_train_set.txt'
    qid_list, tid_list = load_question_topic_set(topic_info_fp)

    tid_rate = dict()
    for tids in tid_list:
        for tid in tids:
            tid_rate[tid] = tid_rate.get(tid, 0.) + 1.
    for tid in tid_rate:
        tid_rate[tid] /= (1. * len(tid_list))

    # load hash table of label
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    features = [0.] * 1999
    for tid in tid_rate:
        features[int(label2id[tid])] = tid_rate[tid]

    feature_file_path = '%s/topic_fs_rate.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'), 'all')
    feature_file = open(feature_file_path, 'w')
    feature_file.write('%d 1\n' % len(features))
    for feature in features:
        Feature.save_feature([feature], feature_file)
    feature_file.close()


def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)
