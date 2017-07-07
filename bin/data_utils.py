#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 21:39
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

import ConfigParser
from utils import DataUtil
import sys
import json
import math


def load_question_set(fp):
    """
    load `question_train_set.txt` and `question_eval_set.txt`
    :param fp:
    :return:
    """
    f = open(fp)
    qid_list = []
    tc_list = []
    tw_list = []
    dc_list = []
    dw_list = []
    index = 0
    for line in f:
        subs = line.strip('\n').split('\t')
        qid_list.append(subs[0])
        tc_list.append(subs[1].split(','))
        tw_list.append(subs[2].split(','))
        dc_list.append(subs[3].split(','))
        dw_list.append(subs[4].split(','))
        index += 1
    f.close()
    return qid_list, tc_list, tw_list, dc_list, dw_list


def load_topic_info(fp):
    f = open(fp)
    tid_list = []
    father_list = []
    tc_list = []
    tw_list = []
    dc_list = []
    dw_list = []
    for line in f:
        subs = line.strip('\n').split('\t')
        tid_list.append(subs[0])
        father_list.append(subs[1].split(','))
        tc_list.append(subs[2].split(','))
        tw_list.append(subs[3].split(','))
        dc_list.append(subs[4].split(','))
        dw_list.append(subs[5].split(','))
    f.close()
    return tid_list, father_list, tc_list, tw_list, dc_list, dw_list


def load_question_topic_set(fp):
    """
    load file `question_topic_train_set.txt`
    :param fp:
    :return:
    """
    qid_list = []
    tid_list = []

    f = open(fp)
    for line in f:
        subs = line.strip('\n').split('\t')
        qid_list.append(subs[0])
        tid_list.append(subs[1].split(','))
    f.close()
    return qid_list, tid_list


def random_split_dataset(config):
    all_fp = config.get('DIRECTORY', 'dataset_pt') + 'title_content_word.all.csv'
    all_data = open(all_fp, 'r').readlines()
    all_data = [line.strip('\n') for line in all_data]
    [train, valid] = DataUtil.random_split(all_data, [0.966, 0.034])
    train_fp = config.get('DIRECTORY', 'dataset_pt') + 'title_content_word.train_996.csv'
    valid_fp = config.get('DIRECTORY', 'dataset_pt') + 'title_content_word.valid_034.csv'
    DataUtil.save_vector(train_fp, train, 'w')
    DataUtil.save_vector(valid_fp, valid, 'w')


def random_split_index_offline(config):
    question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    question_offline = open(question_offline_fp, 'r').readlines()
    [train, valid] = DataUtil.random_split(range(len(question_offline)), [0.966, 0.034])
    train_fp = config.get('DIRECTORY', 'index_pt') + 'train_996.offline.index'
    valid_fp = config.get('DIRECTORY', 'index_pt') + 'valid_034.offline.index'
    DataUtil.save_vector(train_fp, train, 'w')
    DataUtil.save_vector(valid_fp, valid, 'w')


def generate_title_doc_char_dataset(config):
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    question_train_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    qid_train, tc_train, tw_train, dc_train, dw_train = load_question_set(question_train_fp)

    topic_train_fp = config.get('DIRECTORY', 'source_pt') + '/question_topic_train_set.txt'
    qid_train, tid_train = load_question_topic_set(topic_train_fp)

    title_content_char_fp = config.get('DIRECTORY', 'dataset_pt') + '/title_content_char.offline.csv'
    title_content_char = open(title_content_char_fp, 'w')
    for line_id in range(len(qid_train)):
        line = '%s\t%s\t%s\t%s\n' % (qid_train[line_id],
                                     ','.join(tc_train[line_id]),
                                     ','.join(dc_train[line_id]),
                                     ','.join([str(label2id[label]) for label in tid_train[line_id]]))
        title_content_char.write(line)
    title_content_char.close()

    question_online_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
    qid_online, tc_online, tw_online, dc_online, dw_online = load_question_set(question_online_fp)

    title_content_char_fp = config.get('DIRECTORY', 'dataset_pt') + '/title_content_char.online.csv'
    title_content_char = open(title_content_char_fp, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\t%s\t%s\t\n' % (qid_online[line_id],
                                   ','.join(tc_online[line_id]),
                                   ','.join(dc_online[line_id]))
        title_content_char.write(line)
    title_content_char.close()


def generate_dataset(config):
    label2id_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'label2id_fn'))
    label2id = json.load(open(label2id_fp, 'r'))

    question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    qid_offline, tc_offline, tw_offline, dc_offline, dw_offline = load_question_set(question_offline_fp)

    topic_train_fp = config.get('DIRECTORY', 'source_pt') + '/question_topic_train_set.txt'
    qid_offline, tid_offline = load_question_topic_set(topic_train_fp)

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/question_id.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % qid_offline[line_id]
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_char.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % ','.join(tc_offline[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_word.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % ','.join(tw_offline[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_char.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % ','.join(dc_offline[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_word.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % ','.join(dw_offline[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/label_id.offline.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_offline)):
        line = '%s\n' % ','.join([str(label2id[label]) for label in tid_offline[line_id]])
        f.write(line)
    f.close()

    question_online_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
    qid_online, tc_online, tw_online, dc_online, dw_online = load_question_set(question_online_fp)

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/question_id.online.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\n' % qid_online[line_id]
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_char.online.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\n' % ','.join(tc_online[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_word.online.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\n' % ','.join(tw_online[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_char.online.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\n' % ','.join(dc_online[line_id])
        f.write(line)
    f.close()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_word.online.csv'
    f = open(file_path, 'w')
    for line_id in range(len(qid_online)):
        line = '%s\n' % ','.join(dw_online[line_id])
        f.write(line)
    f.close()


def generate_idf(config):
    question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    qid_offline, tc_offline, tw_offline, dc_offline, dw_offline = load_question_set(question_offline_fp)
    question_online_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
    qid_online, tc_online, tw_online, dc_online, dw_online = load_question_set(question_online_fp)

    word_idf = {}
    for i in range(len(qid_offline)):
        for word in set(tw_offline[i] + dw_offline[i]):
            word_idf[word] = word_idf.get(word, 0) + 1
    for i in range(len(qid_online)):
        for word in set(tw_online[i] + dw_online[i]):
            word_idf[word] = word_idf.get(word, 0) + 1
    tol_num = len(qid_offline) + len(qid_online)
    for word in word_idf:
        word_idf[word] = math.log(tol_num / (word_idf[word] + 1.)) / math.log(2.)

    word_idf_fp = config.get('DIRECTORY', 'stat_pt') + 'word_idf.txt'
    word_idf = ['%s\t%s' % (str(kv[0]), str(kv[1])) for kv in sorted(word_idf.items(), lambda x, y: cmp(x[1], y[1]))]
    DataUtil.save_vector(word_idf_fp, word_idf, 'w')


def _test_load_question_set(cf):
    q_train_set = cf.get('DEFAULT', 'source_pt') + '/question_train_set.txt.small'

    (qid_list, tc_list, tw_list, dc_list, dw_list) = load_question_set(q_train_set)
    print qid_list
    print tc_list
    print tw_list
    print dc_list
    print dw_list


def _test_load_topic_info(cf):
    q_topic_set = cf.get('DEFAULT', 'source_pt') + '/topic_info.txt.small'

    (tid_list, father_list, tc_list, tw_list, dc_list, dw_list) = load_topic_info(q_topic_set)
    print tid_list
    print father_list
    print tc_list
    print tw_list
    print dc_list
    print dw_list


def _test():
    conf_fp = '/Users/houjianpeng/Github/zhihu-machine-learning-challenge-2017/conf/default.conf'
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)

    # _test_load_question_set(cf)
    _test_load_topic_info(cf)


def main():
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    random_split_index_offline(config)


if __name__ == '__main__':
    # _test()
    main()
