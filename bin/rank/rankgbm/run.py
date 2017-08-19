# -*- coding: utf-8 -*-
# ! /usr/bin/python

import sys

reload(sys)
sys.path.append("..")
sys.setdefaultencoding('utf-8')

import numpy as np
import json
import ConfigParser
import sys
from ...utils import LogUtil, DataUtil
from ...text_cnn.data_helpers import load_labels_from_file
from ...evaluation import F_by_ids
import hashlib
from rankgbm import RankGBM
import time
import os
import math


def get_all_feature_names(config):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    # load feture names
    model_feature_names = config.get('RANK', 'model_features').split()
    model_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                           model_feature_names]

    instance_feature_names = config.get('RANK', 'instance_features').split()
    instance_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                              instance_feature_names]

    topic_feature_names = config.get('RANK', 'topic_features').split()
    topic_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                           topic_feature_names]

    all_feature_names = list()
    for fn in (model_feature_names + instance_feature_names + topic_feature_names):
        if len(fn) and (fn not in all_feature_names):
            all_feature_names.append(fn)

    return all_feature_names


def init_out_dir(config, out_tag):
    # generate output tag
    # out_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    config.set('DIRECTORY', 'out_tag', str(out_tag))
    # generate output directory
    out_pt = config.get('DIRECTORY', 'out_pt')
    out_pt_exists = os.path.exists(out_pt)
    if out_pt_exists:
        LogUtil.log("WARNING", 'out path (%s) already exists ' % out_pt)
    else:
        os.mkdir(out_pt)
        os.mkdir(config.get('DIRECTORY', 'pred_pt'))
        os.mkdir(config.get('DIRECTORY', 'model_pt'))
        os.mkdir(config.get('DIRECTORY', 'conf_pt'))
        os.mkdir(config.get('DIRECTORY', 'score_pt'))
        LogUtil.log('INFO', 'out path (%s) created ' % out_pt)
    # save config
    config.write(open(config.get('DIRECTORY', 'conf_pt') + 'featwheel.conf', 'w'))


def load_rank_file(file_path):
    '''
    加载排序数据，数据格式：'rank qid:$qid $feature_id:$feature_value ...'
    '''
    instances = []
    file = open(file_path)
    for line in file:
        line = line.split('#', 1)[0]
        subs = filter(None, line.split(' '))
        rank = int(subs[0])
        [k, v] = subs[1].split(':')
        assert k == 'qid', 'do not find query id'
        qid = int(v)
        features = []
        for i in range(2, len(subs)):
            [k, v] = subs[i].split(':')
            fid = int(k)
            fvalue = float(v)
            if math.isnan(fvalue) or math.isinf(fvalue):
                fvalue = 0.
            while len(features) <= fid:
                features.append(0.0)
            features[fid] = fvalue
        instances.append([rank, qid, features])
    file.close()
    LogUtil.log("INFO", "load rank file done. N(instances)=%d, %s" % (len(instances), file_path))
    return instances


def train(config, argv):
    fold_id = int(argv[0])
    out_tag = argv[1]
    init_out_dir(config, out_tag)

    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    all_feature_names = get_all_feature_names(config)

    feature_names_md5 = hashlib.md5('|'.join(all_feature_names)).hexdigest()
    LogUtil.log('INFO', 'feature_names_md5=%s' % feature_names_md5)

    # load rank train + valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load labels
    all_valid_labels = load_labels_from_file(config, 'offline', valid_index).tolist()

    num_instance = 100000
    num_p1 = 33333
    num_p2 = 33333
    num_p3 = num_instance - num_p1 - num_p2

    # generate index for instance
    ins_p1_indexs = [i for i in range(num_p1)]
    ins_p2_indexs = [i for i in range(num_p1, num_p1 + num_p2)]
    ins_p3_indexs = [i for i in range(num_p1 + num_p2, num_instance)]

    # generate index for each part
    rank_p1_indexs = [i for i in range(num_p1 * vote_k)]
    rank_p2_indexs = [(num_p1 * vote_k + i) for i in range(num_p2 * vote_k)]
    rank_p3_indexs = [((num_p1 + num_p2) * vote_k + i) for i in range(num_p3 * vote_k)]

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    train_file_name = '%s/featwheel_vote_%d_%s.fold%d_train.rank' % (config.get('DIRECTORY', 'dataset_pt'),
                                                                     vote_k,
                                                                     feature_names_md5,
                                                                     fold_id)
    train_instances = load_rank_file(train_file_name)

    valid_file_name = '%s/featwheel_vote_%d_%s.fold%d_valid.rank' % (config.get('DIRECTORY', 'dataset_pt'),
                                                                     vote_k,
                                                                     feature_names_md5,
                                                                     fold_id)
    valid_instances = load_rank_file(valid_file_name)

    valid_Xs = np.array([valid_instances[i][2] for i in range(len(valid_instances))])

    # 训练模型
    rank_gbm = RankGBM(vote_k,
                       n_round=config.getint('RANK_GBM', 'n_round'),
                       max_depth=config.getint('RANK_GBM', 'max_depth'),
                       max_features=float(config.get('RANK_GBM', 'max_features')),
                       min_samples_leaf=config.getint('RANK_GBM', 'min_samples_leaf'),
                       learn_rate=float(config.get('RANK_GBM', 'learn_rate')))
    rank_gbm.fit(train_instances, {"vali": valid_instances})
    # 对预测数据进行预测
    valid_preds = rank_gbm.predict(valid_Xs)
    valid_preds = zip(*[iter(valid_preds)] * vote_k)

    if 0 == fold_id:
        ins_indexs = ins_p3_indexs
    elif 1 == fold_id:
        ins_indexs = ins_p1_indexs
    elif 2 == fold_id:
        ins_indexs = ins_p2_indexs
    else:
        ins_indexs = None

    valid_preds_ids = list()
    valid_vote_k_label = [vote_k_label[iid] for iid in ins_indexs]
    for i in range(len(valid_vote_k_label)):
        valid_preds_ids.append(
            [kv[0] for kv in sorted(zip(valid_vote_k_label[i], valid_preds[i]), key=lambda x: x[1], reverse=True)])
    valid_labels = [all_valid_labels[iid] for iid in ins_indexs]

    LogUtil.log('INFO', '------------ fold score ---------------')
    F_by_ids(valid_preds_ids, valid_labels)

    model_file_path = config.get('DIRECTORY', 'model_pt') + '/rankgbm_fold%d' % fold_id
    rank_gbm.save(model_file_path)


def test(config, argv):
    model0 = RankGBM.load(config.get('DIRECTORY', 'model_pt') + '/rankgbm_fold%d' % 0)
    model1 = RankGBM.load(config.get('DIRECTORY', 'model_pt') + '/rankgbm_fold%d' % 1)
    model2 = RankGBM.load(config.get('DIRECTORY', 'model_pt') + '/rankgbm_fold%d' % 2)
    predict_online(config, [model0, model1, model2])


def predict_online(config, models):
    version_id = config.get('RANK', 'version_id')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    all_feature_names = get_all_feature_names(config)

    feature_names_md5 = hashlib.md5('|'.join(all_feature_names)).hexdigest()
    LogUtil.log('INFO', 'feature_names_md5=%s' % feature_names_md5)

    valid_preds = [0, 0, 0]
    for fold_id in range(3):
        valid_file_name = '%s/featwheel_vote_%d_%s.fold%d_valid.rank' % (config.get('DIRECTORY', 'dataset_pt'),
                                                                         vote_k,
                                                                         feature_names_md5,
                                                                         fold_id)
        valid_instances = load_rank_file(valid_file_name)

        valid_Xs = np.array([valid_instances[i][2] for i in range(len(valid_instances))])
        valid_preds[fold_id] = models[fold_id].predict(valid_Xs)
    valid_preds = valid_preds[1] + valid_preds[2] + valid_preds[0]
    valid_preds = zip(*[iter(valid_preds)] * vote_k)

    # load rank train + valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    # load labels
    all_valid_labels = load_labels_from_file(config, 'offline', valid_index).tolist()

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append(
            [kv[0] for kv in sorted(zip(vote_k_label[i], valid_preds[i]), key=lambda x: x[1], reverse=True)])

    LogUtil.log('INFO', '------------ vote score ---------------')
    F_by_ids(vote_k_label, all_valid_labels)
    LogUtil.log('INFO', '------------ rank score ---------------')
    F_by_ids(preds_ids, all_valid_labels)

    valid_preds_file = open('%s/rank_id_score.validation.%s' % (config.get('DIRECTORY', 'pred_pt'), version_id), 'w')
    for i in range(len(vote_k_label)):
        valid_preds_file.write(
            ','.join(['%s:%s' % (kv[0], kv[1]) for kv in zip(vote_k_label[i], valid_preds[i])]) + '\n')
    valid_preds_file.close()

    test_file_name = '%s/featwheel_vote_%d_%s.test.rank' % (config.get('DIRECTORY', 'dataset_pt'),
                                                            vote_k,
                                                            feature_names_md5)
    test_instances = load_rank_file(test_file_name)
    test_Xs = np.array([test_instances[i][2] for i in range(len(test_instances))])

    test_preds1 = models[0].predict(test_Xs)
    test_preds2 = models[1].predict(test_Xs)
    test_preds3 = models[2].predict(test_Xs)

    test_preds = [(test_preds1[line_id] + test_preds2[line_id] + test_preds3[line_id]) / 3. for line_id in
                  range(len(test_preds1))]
    test_preds = zip(*[iter(test_preds)] * vote_k)

    # load topk ids
    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'online')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append(
            [kv[0] for kv in sorted(zip(vote_k_label[i], test_preds[i]), key=lambda x: x[1], reverse=True)])

    # load question ID for online dataset
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')

    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))

    rank_submit_fp = '%s/rank_submit.online.%s' % (config.get('DIRECTORY', 'pred_pt'), version_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for line_id, p in enumerate(preds_ids):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_f.close()

    rank_submit_fp = '%s/rank_id_score.online.%s' % (config.get('DIRECTORY', 'pred_pt'), version_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for i in range(len(vote_k_label)):
        rank_submit_f.write(','.join(['%s:%s' % (kv[0], kv[1]) for kv in zip(vote_k_label[i], test_preds[i])]) + '\n')
    rank_submit_f.close()

    rank_submit_ave_fp = '%s/rank_ave.online.%s' % (config.get('DIRECTORY', 'pred_pt'), version_id)
    rank_submit_ave_f = open(rank_submit_ave_fp, 'w')
    for line_id, p in enumerate(vote_k_label):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_ave_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_ave_f.close()


def ave(config, argv):
    valid_row_size = 100000
    valid_col_size = 10
    valid_preds = list()
    for i in range(valid_row_size):
        vec = [0.] * valid_col_size
        valid_preds.append(vec)

    paths = argv[0]
    paths = paths.split(',')
    for path in paths:
        f = open(path, 'r')
        lid = 0
        for line in f:
            subs = line.split(',')
            for pid, kv in enumerate(subs):
                valid_preds[lid][pid] += float(kv.split(':')[1])
            lid += 1
        f.close()

    version_id = config.get('RANK', 'version_id')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    # load rank train + valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    # load labels
    all_valid_labels = load_labels_from_file(config, 'offline', valid_index).tolist()

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append(
            [kv[0] for kv in sorted(zip(vote_k_label[i], valid_preds[i]), key=lambda x: x[1], reverse=True)])

    LogUtil.log('INFO', '------------ vote score ---------------')
    F_by_ids(vote_k_label, all_valid_labels)
    LogUtil.log('INFO', '------------ rank score ---------------')
    F_by_ids(preds_ids, all_valid_labels)


def tmp(config, argv):
    version_id = config.get('RANK', 'version_id')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'online')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    pre_fp = '%s/rank_all.online.%s' % (config.get('DIRECTORY', 'pred_pt'), version_id)
    test_preds = DataUtil.load_matrix(pre_fp, 'float')

    rank_submit_fp = '%s/rank_id_score.online.%s.bak' % (config.get('DIRECTORY', 'pred_pt'), version_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for i in range(len(vote_k_label)):
        rank_submit_f.write(','.join(['%s:%s' % (kv[0], kv[1]) for kv in zip(vote_k_label[i], test_preds[i])]) + '\n')
    rank_submit_f.close()


if __name__ == "__main__":
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
