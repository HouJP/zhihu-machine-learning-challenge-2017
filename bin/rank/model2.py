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


def train(config, argv):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    # load feture names
    feature_names = config.get('RANK', 'model_features').split()
    feature_names = ['featwheel_vote_%d_%s_%s' % (vote_k, vote_k_label_file_name, fn) for fn in feature_names]

    # load feature matrix
    offline_features = Feature.load_all(config.get('DIRECTORY', 'dataset_pt'),
                                        feature_names,
                                        'offline',
                                        False)

    # load labels
    offline_labels_file_path = '%s/featwheel_vote_%d_%s.%s.label' % (config.get('DIRECTORY', 'label_pt'),
                                                                  vote_k,
                                                                     vote_k_label_file_name,
                                                                  'offline')
    offline_labels = DataUtil.load_vector(offline_labels_file_path, 'int')

    # generete indexs
    rank_train_indexs = [i for i in range(50000 * vote_k)]
    rank_valid_indexs = [(50000 * vote_k + i) for i in range(50000 * vote_k)]

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtrain.set_group([vote_k] * (len(train_labels) / vote_k))

    dvalid = xgb.DMatrix(valid_features, label=valid_labels)
    dvalid.set_group([vote_k] * (len(valid_labels) / vote_k))

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    params = load_parameters(config)
    model = xgb.train(params,
                      dtrain,
                      params['num_round'],
                      watchlist,
                      early_stopping_rounds=params['early_stop'],
                      verbose_eval=params['verbose_eval'])
    LogUtil.log('INFO', 'best_ntree_limit=%d' % model.best_ntree_limit)

    # load valid dataset index
    valid_index_fp = '%s/%s.offline.index' % (config.get('DIRECTORY', 'index_pt'),
                                              config.get('TITLE_CONTENT_CNN', 'valid_index_offline_fn'))
    valid_index = DataUtil.load_vector(valid_index_fp, 'int')
    valid_index = [num - 1 for num in valid_index]

    # load labels
    valid_labels = load_labels_from_file(config, 'offline', valid_index).tolist()[50000:]
    # make prediction
    # valid_preds = model.predict(dvalid)
    valid_preds = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    # valid_preds = model.predict(dvalid, ntree_limit=params['num_round'])
    valid_preds = [num for num in valid_preds]
    valid_preds = zip(*[iter(valid_preds)] * vote_k)

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_k_label_fp = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
    vote_k_label = DataUtil.load_matrix(vote_k_label_fp, 'int')[50000:]

    preds_ids = list()
    for i in range(len(vote_k_label)):
        preds_ids.append([kv[0] for kv in sorted(zip(vote_k_label[i], valid_preds[i]), key=lambda x:x[1], reverse=True)])

    F_by_ids(vote_k_label, valid_labels)
    F_by_ids(preds_ids, valid_labels)

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


def predict_online(model, best_ntree_limit):
    run_id = config.get('RANK', 'rank_id')

    dtest_fp = stand_path('%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'dmatrix', run_id, 'online'))
    group_test_fp = '%s/rank_%s_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'group', run_id, 'online')
    dtest = xgb.DMatrix(dtest_fp)
    dtest.set_group(DataUtil.load_vector(group_test_fp, 'int'))

    # make prediction
    topk = config.getint('RANK', 'topk')
    test_preds = model.predict(dtest, ntree_limit=best_ntree_limit)
    test_preds = [num for num in test_preds]
    test_preds = zip(*[iter(test_preds)] * topk)

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    topk_class_index_fp = '%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), 'online')
    topk_label_id = DataUtil.load_matrix(topk_class_index_fp, 'int')

    preds_ids = list()
    for i in range(len(topk_label_id)):
        preds_ids.append([kv[0] for kv in sorted(zip(topk_label_id[i], test_preds[i]), key=lambda x:x[1], reverse=True)])

    # load question ID for online dataset
    qid_on_fp = '%s/%s.online.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'question_id')
    qid_on = DataUtil.load_vector(qid_on_fp, 'str')
    LogUtil.log('INFO', 'load online question ID done')

    # load hash table of label
    id2label_fp = '%s/%s' % (config.get('DIRECTORY', 'hash_pt'), config.get('TITLE_CONTENT_CNN', 'id2label_fn'))
    id2label = json.load(open(id2label_fp, 'r'))

    rank_submit_fp = '%s/rank_submit.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), run_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for line_id, p in enumerate(preds_ids):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_f.close()

    rank_submit_fp = '%s/rank_all.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), run_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for p in test_preds:
        rank_submit_f.write('%s\n' % ','.join([str(num) for num in p]))
    rank_submit_f.close()

    rank_submit_ave_fp = '%s/rank_ave.online.%s' % (config.get('DIRECTORY', 'tmp_pt'), run_id)
    rank_submit_ave_f = open(rank_submit_ave_fp, 'w')
    for line_id, p in enumerate(topk_label_id):
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
