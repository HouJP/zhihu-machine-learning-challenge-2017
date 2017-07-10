#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 21:39
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

import ConfigParser
from utils import DataUtil, LogUtil
import sys
import json
import math
import multiprocessing


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
        subs = line.strip().split('\t')
        tid_list.append(subs[0])
        if 1 < len(subs):
            father_list.append(subs[1].split(','))
        else:
            father_list.append([])
        if 2 < len(subs):
            tc_list.append(subs[2].split(','))
        else:
            tc_list.append([])
        if 3 < len(subs):
            tw_list.append(subs[3].split(','))
        else:
            tw_list.append([])
        if 4 < len(subs):
            dc_list.append(subs[4].split(','))
        else:
            dc_list.append([])
        if 5 < len(subs):
            dw_list.append(subs[5].split(','))
        else:
            dw_list.append([])
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


def load_idf(file_path):
    idf = {}
    f = open(file_path)
    for line in f:
        word, word_idf = line.strip('\n').split('\t')
        idf[word] = float(word_idf)
    f.close()
    return idf


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

    char_idf = {}
    for i in range(len(qid_offline)):
        for char in set(tc_offline[i] + dc_offline[i]):
            char_idf[char] = char_idf.get(char, 0) + 1
    for i in range(len(qid_online)):
        for char in set(tc_online[i] + dc_online[i]):
            char_idf[char] = char_idf.get(char, 0) + 1
    tol_num = len(qid_offline) + len(qid_online)
    for char in char_idf:
        char_idf[char] = math.log(tol_num / (char_idf[char] + 1.)) / math.log(2.)

    char_idf_fp = config.get('DIRECTORY', 'stat_pt') + 'char_idf.txt'
    char_idf = ['%s\t%s' % (str(kv[0]), str(kv[1])) for kv in sorted(char_idf.items(), lambda x, y: cmp(x[1], y[1]))]
    DataUtil.save_vector(char_idf_fp, char_idf, 'w')


def idf_filter(words, idf, min_idf, max_idf):
    return [word for word in words if min_idf < idf[word] < max_idf]


def generate_single_idf_dataset(file_path, qid, docs, idf):
    max_len = 0
    min_len = sys.maxint
    ave_len = 0

    tol_num = 2999967. + 217360.
    min_idf = math.log(tol_num / (200000. + 1.)) / math.log(2.)
    max_idf = math.log(tol_num / (40. + 1.)) / math.log(2.)
    LogUtil.log('INFO', 'min_idf=%s, max_idf=%s' % (str(min_idf), str(max_idf)))

    f = open(file_path, 'w')
    for line_id in range(len(qid)):
        vec = idf_filter(docs[line_id], idf, min_idf, max_idf)
        line = '%s\n' % ','.join(vec)
        f.write(line)
        vec_len = len(vec)
        max_len = max(max_len, vec_len)
        min_len = min(min_len, vec_len)
        ave_len += vec_len
    ave_len /= (1. * len(qid))
    f.close()
    LogUtil.log('INFO', 'generate_single_tfidf_dataset (%s) done' % file_path)
    LogUtil.log('INFO', 'max_len=%d' % max_len)
    LogUtil.log('INFO', 'min_len=%d' % min_len)
    LogUtil.log('INFO', 'ave_len=%d' % ave_len)


def generate_idf_dataset(config):

    word_idf_fp = config.get('DIRECTORY', 'stat_pt') + 'word_idf.txt'
    word_idf = load_idf(word_idf_fp)
    char_idf_fp = config.get('DIRECTORY', 'stat_pt') + 'char_idf.txt'
    char_idf = load_idf(char_idf_fp)

    question_offline_fp = config.get('DIRECTORY', 'source_pt') + '/question_train_set.txt'
    qid_offline, tc_offline, tw_offline, dc_offline, dw_offline = load_question_set(question_offline_fp)

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_char_idf.offline.csv'
    processor_tc_offline = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_offline, tc_offline, char_idf))
    processor_tc_offline.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_word_idf.offline.csv'
    processor_tw_offline = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_offline, tw_offline, word_idf))
    processor_tw_offline.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_char_idf.offline.csv'
    processor_cc_offline = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_offline, dc_offline, char_idf))
    processor_cc_offline.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_word_idf.offline.csv'
    processor_cw_offline = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_offline, dw_offline, word_idf))
    processor_cw_offline.start()

    question_online_fp = config.get('DIRECTORY', 'source_pt') + '/question_eval_set.txt'
    qid_online, tc_online, tw_online, dc_online, dw_online = load_question_set(question_online_fp)

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_char_idf.online.csv'
    processor_tc_online = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_online, tc_online, char_idf))
    processor_tc_online.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/title_word_idf.online.csv'
    processor_tw_online = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_online, tw_online, word_idf))
    processor_tw_online.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_char_idf.online.csv'
    processor_cc_online = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_online, dc_online, char_idf))
    processor_cc_online.start()

    file_path = config.get('DIRECTORY', 'dataset_pt') + '/content_word_idf.online.csv'
    processor_cw_online = multiprocessing.Process(target=generate_single_idf_dataset, args=(
        file_path, qid_online, dw_online, word_idf))
    processor_cw_online.start()

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))


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
