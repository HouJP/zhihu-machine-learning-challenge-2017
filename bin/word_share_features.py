#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/18 09:29
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
from data_utils import load_topic_info, load_question_set


def save_word_share_features(config, dataset_name, tw_list):
    if 'offline' == dataset_name:
        question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    elif 'online' == dataset_name:
        question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
    else:
        question_offline_fp = None
    qid_all, tc_all, tw_all, dc_all, dw_all = load_question_set(question_offline_fp)
    features = [[0.] * 1999] * len(qid_all)
    for line_id in range(len(tw_all)):
        for word in tw_all[line_id]:
            for topic_id in range(1999):
                if word in tw_list[topic_id]:
                    features[line_id][topic_id] += 1

    ws_fs_f = open('%s/word_share.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), dataset_name), 'w')
    for feature in features:
        ws_fs_f.write('%s\n' % ' '.join([str(num) for num in feature]))
    ws_fs_f.close()


def generate_word_share_features(config, argv):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

    save_word_share_features(config, 'offline', tw_list)
    save_word_share_features(config, 'online', tw_list)


def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)
