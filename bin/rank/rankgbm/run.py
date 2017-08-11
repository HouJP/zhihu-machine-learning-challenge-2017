# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys

reload(sys)
sys.path.append("..")
sys.setdefaultencoding('utf-8')

import numpy as np  
import ConfigParser
import sys
from ...utils import LogUtil, DataUtil
from ...text_cnn.data_helpers import load_labels_from_file
from ...evaluation import F_by_ids
import hashlib
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


def train(config, params):
    '''
    针对`target_param`进行grid_search
    '''
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

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
    valid_preds1 = rank_gbm.predict(valid_Xs)

    preds_ids1 = list()
    vote_k_label1 = [vote_k_label[iid] for iid in ins_p3_indexs]
    for i in range(len(vote_k_label1)):
        preds_ids1.append(
            [kv[0] for kv in sorted(zip(vote_k_label1[i], valid_preds1[i]), key=lambda x: x[1], reverse=True)])
    valid_labels1 = [all_valid_labels[iid] for iid in ins_p3_indexs]

    LogUtil.log('INFO', '------------ fold1 score ---------------')
    F_by_ids(preds_ids1, valid_labels1)


if __name__ == "__main__":
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    params = {'n_round' : 10, 'max_depth' : 7, 'max_features' : "auto", 'min_samples_leaf' : 0.025, 'learn_rate' : 1.0, 'silent' : False}
    train(config, params)
