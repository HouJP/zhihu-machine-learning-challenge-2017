#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/30 00:33
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import xgboost as xgb


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


def train(config, argv):
    dtrain_train_fp = '%s/%s_train.libsvm' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name'))
    group_train_fp = '%s/%s_train.group' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name'))

    dtrain_valid_fp = '%s/%s_valid.libsvm' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name'))
    group_valid_fp = '%s/%s_valid.group' % (config.get('DIRECTORY', 'dataset_pt'), config.get('RANK', 'dmatrix_name'))

    dtrain = xgb.DMatrix(dtrain_train_fp)
    dtrain.set_group(group_train_fp)

    dvalid = xgb.DMatrix(dtrain_valid_fp)
    dvalid.set_group(group_valid_fp)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    params = load_parameters(config)
    model = xgb.train(params,
                      dtrain,
                      params['num_round'],
                      watchlist,
                      early_stopping_rounds=params['early_stop'],
                      verbose_eval=params['verbose_eval'])
