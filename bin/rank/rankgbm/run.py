# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys

reload(sys)
sys.path.append("..")
sys.setdefaultencoding('utf-8')

import numpy as np  

from ...utils import LogUtil
from rankgbm import RankGBM


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
            while len(features) <= fid:
                features.append(0.0)
            features[fid] = fvalue
        instances.append([rank, qid, features])
    file.close()
    LogUtil.log("INFO", "load rank file done. N(instances)=%d" % (len(instances)))
    return instances


def train(params):
    '''
    针对`target_param`进行grid_search
    '''
    train_file_name = '/mnt/disk2/xinyu/data/dataset/featwheel_vote_10_fe90ef2ad1a5f75899b6653ce822831b.fold0_train.rank'
    train_instances = load_rank_file(train_file_name)

    valid_file_name = '/mnt/disk2/xinyu/data/dataset/featwheel_vote_10_fe90ef2ad1a5f75899b6653ce822831b.fold0_valid.rank'
    valid_instances = load_rank_file(valid_file_name)

    valid_Xs = np.array([ valid_instances[i][2] for i in range (len(valid_instances))])

    # 训练模型
    rank_gbm = RankGBM(n_round = params['n_round'],
        max_depth = params['max_depth'],
        max_features = params['max_features'],
        min_samples_leaf = params['min_samples_leaf'],
        learn_rate = params['learn_rate'],
        silent = params['silent'])
    rank_gbm.fit(train_instances, {"vali": valid_instances})
    # 对预测数据进行预测
    valid_preds = rank_gbm.predict(valid_Xs)


if __name__ == "__main__":

    params = {'n_round' : 400, 'max_depth' : 7, 'max_features' : "auto", 'min_samples_leaf' : 0.025, 'learn_rate' : 1.0, 'silent' : False}
    train(params)
