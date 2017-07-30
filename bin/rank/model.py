#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/30 00:33
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import xgboost as xgb
import sys
import ConfigParser
import json
from ..utils import DataUtil, LogUtil
from ..text_cnn.data_helpers import load_labels_from_file
from ..evaluation import F_by_ids


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
    # params['alpha'] = float(config.get('XGB_PARAMS', 'alpha'))
    # params['lambda'] = float(config.get('XGB_PARAMS', 'lambda'))
    params['verbose_eval'] = config.getint('XGB_PARAMS', 'verbose_eval')
    return params


def stand_path(s):
    return '/' + '/'.join(filter(None, s.split('/')))


def train(config, argv):
    dtrain_fp = stand_path('%s/%s_train.libsvm' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name')))
    group_train_fp = dtrain_fp + '.group'
    dtrain = xgb.DMatrix(dtrain_fp)
    dtrain.set_group(DataUtil.load_vector(group_train_fp, 'int'))

    dvalid_fp = stand_path('%s/%s_valid.libsvm' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name')))
    group_valid_fp = dvalid_fp + '.group'
    dvalid = xgb.DMatrix(dvalid_fp)
    dvalid.set_group(DataUtil.load_vector(group_valid_fp, 'int'))

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
    topk = config.getint('RANK', 'topk')
    valid_preds = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    valid_preds = [num for num in valid_preds]
    valid_preds = zip(*[iter(valid_preds)] * topk)

    # load topk ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    topk_class_index_fp = '%s/%s.%s.index' % (index_pt, config.get('RANK', 'topk_class_index'), 'offline')
    topk_label_id = DataUtil.load_matrix(topk_class_index_fp, 'int')[50000:]

    preds_ids = list()
    for i in range(50000):
        preds_ids.append([kv[0] for kv in sorted(zip(topk_label_id[i], valid_preds[i]), key=lambda x:x[1], reverse=True)])

    F_by_ids(preds_ids, valid_labels)


def train_online(config, argv):
    dtrain_fp = stand_path('%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'dmatrix', 'offline'))
    group_train_fp = '%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'group', 'offline')
    dtrain = xgb.DMatrix(dtrain_fp)
    dtrain.set_group(DataUtil.load_vector(group_train_fp, 'int'))

    dtest_fp = stand_path('%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'dmatrix', 'online'))
    group_test_fp = '%s/rank_%s.%s.csv' % (config.get('DIRECTORY', 'dataset_pt'), 'group', 'online')
    dtest = xgb.DMatrix(dtest_fp)
    dtest.set_group(DataUtil.load_vector(group_test_fp, 'int'))

    params = load_parameters(config)
    model = xgb.train(params,
                      dtrain,
                      params['num_round'])
    LogUtil.log('INFO', 'best_ntree_limit=%d' % model.best_ntree_limit)

    # make prediction
    topk = config.getint('RANK', 'topk')
    test_preds = model.predict(dtest, ntree_limit=model.best_ntree_limit)
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

    run_id = 2
    rank_submit_fp = '%s/rank_submit.online.%02d' % (config.get('DIRECTORY', 'tmp_pt'), run_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for line_id, p in enumerate(preds_ids):
        label_sorted = [id2label[str(n)] for n in p[:5]]
        rank_submit_f.write("%s,%s\n" % (qid_on[line_id], ','.join(label_sorted)))
        if 0 == line_id % 10000:
            LogUtil.log('INFO', '%d lines prediction done' % line_id)
    rank_submit_f.close()

    rank_submit_fp = '%s/rank_all.online.%02d' % (config.get('DIRECTORY', 'tmp_pt'), run_id)
    rank_submit_f = open(rank_submit_fp, 'w')
    for p in test_preds:
        rank_submit_f.write('%s\n' % ','.join([str(num) for num in p]))
    rank_submit_f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
