#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/30 00:33
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import xgboost as xgb
import hashlib
import sys
import ConfigParser
import json
import numpy as np
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file
from ..evaluation import F_by_ids
from ..featwheel.feature import Feature
from ..featwheel.runner import Runner


def load_parameters(config):
    params = dict()
    params['booster'] = config.get('XGB_PARAMS', 'booster')
    params['objective'] = config.get('XGB_PARAMS', 'objective')
    params['eval_metric'] = config.get('XGB_PARAMS', 'eval_metric')
    params['eta'] = float(config.get('XGB_PARAMS', 'eta'))
    params['max_depth'] = config.getint('XGB_PARAMS', 'max_depth')
    params['subsample'] = float(config.get('XGB_PARAMS', 'subsample'))
    params['colsample_bytree'] = float(config.get('XGB_PARAMS', 'colsample_bytree'))
    params['min_child_weight'] = config.getint('XGB_PARAMS', 'min_child_weight')
    params['silent'] = config.getint('XGB_PARAMS', 'silent')
    params['num_round'] = config.getint('XGB_PARAMS', 'num_round')
    params['early_stop'] = config.getint('XGB_PARAMS', 'early_stop')
    params['nthread'] = config.getint('XGB_PARAMS', 'nthread')
    params['scale_pos_weight'] = float(config.get('XGB_PARAMS', 'scale_pos_weight'))
    params['gamma'] = float(config.get('XGB_PARAMS', 'gamma'))
    params['verbose_eval'] = config.getint('XGB_PARAMS', 'verbose_eval')
    return params


def stand_path(s):
    return '/' + '/'.join(filter(None, s.split('/')))


def self_define_f(preds, dtrain):
    vote_k = config.getint('RANK', 'vote_k')

    labels = zip(*[iter(list(dtrain.get_label()))] * vote_k)
    preds = zip(*[iter(preds)] * vote_k)

    preds_ids = list()
    for i in range(len(preds)):
        preds_ids.append(
            [kv[0] for kv in sorted(enumerate(preds[i]), key=lambda x: x[1], reverse=True)])

    return 'oppsite_f', -1. * F_by_ids(preds_ids, labels)


def train(config, argv):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    # load feture names
    model_feature_names = list(set(config.get('RANK', 'model_features').split()))
    model_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in model_feature_names]

    instance_feature_names = config.get('RANK', 'instance_features').split()
    instance_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in instance_feature_names]

    topic_feature_names = config.get('RANK', 'topic_features').split()
    topic_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in topic_feature_names]

    all_feature_names = [fn for fn in (model_feature_names + instance_feature_names + topic_feature_names) if '' != fn.strip()]

    # pair_feature_names = config.get('RANK', 'pair_features').split()

    # load feature matrix
    offline_features = Feature.load_all(config.get('DIRECTORY', 'dataset_pt'),
                                        all_feature_names,
                                        'offline',
                                        False)

    # load labels
    offline_labels_file_path = '%s/featwheel_vote_%d_%s.%s.label' % (config.get('DIRECTORY', 'label_pt'),
                                                                  vote_k,
                                                                     vote_k_label_file_name,
                                                                  'offline')
    offline_labels = DataUtil.load_vector(offline_labels_file_path, 'int')

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

    # generete indexs
    rank_train_indexs = rank_p1_indexs + rank_p2_indexs
    rank_valid_indexs = rank_p3_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtrain.set_group([vote_k] * (len(train_labels) / vote_k))

    dvalid = xgb.DMatrix(valid_features, label=valid_labels)
    dvalid.set_group([vote_k] * (len(valid_labels) / vote_k))

    valid_preds1, model1 = fit_model(config, dtrain, dvalid)

    # generate indexs
    rank_train_indexs = rank_p2_indexs + rank_p3_indexs
    rank_valid_indexs = rank_p1_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtrain.set_group([vote_k] * (len(train_labels) / vote_k))

    dvalid = xgb.DMatrix(valid_features, label=valid_labels)
    dvalid.set_group([vote_k] * (len(valid_labels) / vote_k))

    valid_preds2, model2 = fit_model(config, dtrain, dvalid)

    # generate indexs
    rank_train_indexs = rank_p3_indexs + rank_p1_indexs
    rank_valid_indexs = rank_p2_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtrain.set_group([vote_k] * (len(train_labels) / vote_k))

    dvalid = xgb.DMatrix(valid_features, label=valid_labels)
    dvalid.set_group([vote_k] * (len(valid_labels) / vote_k))

    valid_preds3, model3 = fit_model(config, dtrain, dvalid)

    valid_preds = valid_preds2 + valid_preds3 + valid_preds1

    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load labels
    valid_labels = load_labels_from_file(config, 'offline', valid_index).tolist()

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append(
            [kv[0] for kv in sorted(zip(vote_k_label[i], valid_preds[i]), key=lambda x: x[1], reverse=True)])

    F_by_ids(vote_k_label, valid_labels)
    F_by_ids(preds_ids, valid_labels)

    predict_online(config, model1, model2, model3)

def fit_model(config, dtrain, dvalid):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    params = load_parameters(config)
    model = xgb.train(params,
                      dtrain,
                      params['num_round'],
                      watchlist,
                      feval=self_define_f,
                      early_stopping_rounds=params['early_stop'],
                      verbose_eval=params['verbose_eval'])
    LogUtil.log('INFO', 'best_ntree_limit=%d' % model.best_ntree_limit)

    # make prediction
    # valid_preds = model.predict(dvalid)
    valid_preds = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    # valid_preds = model.predict(dvalid, ntree_limit=params['num_round'])
    valid_preds = [num for num in valid_preds]
    valid_preds = zip(*[iter(valid_preds)] * vote_k)



    return valid_preds, model

    # predict_online(model, model.best_ntree_limit)
    # predict_online(model, params['num_round'])


def train_online(config, argv):
    rank_id = config.get('RANK', 'rank_id')
    dtrain_fp = stand_path('%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'dmatrix', rank_id, 'offline'))
    group_train_fp = '%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'group', rank_id, 'offline')
    dtrain = xgb.DMatrix(dtrain_fp)
    dtrain.set_group(DataUtil.load_vector(group_train_fp, 'int'))

    params = load_parameters(config)
    model = xgb.train(params,
                      dtrain,
                      params['num_round'])
    LogUtil.log('INFO', 'best_ntree_limit=%d' % model.best_ntree_limit)

    predict_online(model, params['num_round'])


def predict_online(config, model1, model2, model3):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    version_id = config.getint('RANK', 'version_id')

    # load feture names
    model_feature_names = list(set(config.get('RANK', 'model_features').split()))
    model_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                           model_feature_names]

    instance_feature_names = config.get('RANK', 'instance_features').split()
    instance_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                              instance_feature_names]

    topic_feature_names = config.get('RANK', 'topic_features').split()
    topic_feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in
                           topic_feature_names]

    all_feature_names = [fn for fn in (model_feature_names + instance_feature_names + topic_feature_names) if
                         '' != fn.strip()]

    # pair_feature_names = config.get('RANK', 'pair_features').split()

    # load feature matrix
    online_features = Feature.load_all(config.get('DIRECTORY', 'dataset_pt'),
                                        all_feature_names,
                                        'online',
                                        False)

    # load labels
    online_labels_file_path = '%s/featwheel_vote_%d_%s.%s.label' % (config.get('DIRECTORY', 'label_pt'),
                                                                     vote_k,
                                                                     vote_k_label_file_name,
                                                                     'online')
    online_labels = DataUtil.load_vector(online_labels_file_path, 'int')

    test_features, test_labels, _ = Runner._generate_data(range(len(online_labels)), online_labels, online_features, -1)

    dtest = xgb.DMatrix(test_features, label=test_labels)
    dtest.set_group([vote_k] * (len(test_labels) / vote_k))

    # make prediction
    test_preds1 = model1.predict(dtest, ntree_limit=model1.best_ntree_limit)
    test_preds2 = model2.predict(dtest, ntree_limit=model2.best_ntree_limit)
    test_preds3 = model3.predict(dtest, ntree_limit=model3.best_ntree_limit)

    test_preds = [test_preds1[line_id] + test_preds2[line_id] + test_preds3[line_id] for line_id in range(len(test_preds1))]
    test_preds = zip(*[iter(test_preds)] * vote_k)

    # load topk ids
    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'online')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append([kv[0] for kv in sorted(zip(vote_k_label[i], test_preds[i]), key=lambda x:x[1], reverse=True)])

    # load question ID for online dataset
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')

    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))

    rank_submit_fp = '%s/rank_submit.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), version_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for line_id, p in enumerate(preds_ids):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_f.close()

    rank_submit_fp = '%s/rank_all.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), version_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for p in test_preds:
        rank_submit_f.write('%s\n' % ','.join([str(num) for num in p]))
    rank_submit_f.close()

    rank_submit_ave_fp = '%s/rank_ave.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), version_id)
    rank_submit_ave_f = open(rank_submit_ave_fp, 'w')
    for line_id, p in enumerate(vote_k_label):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_ave_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_ave_f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
