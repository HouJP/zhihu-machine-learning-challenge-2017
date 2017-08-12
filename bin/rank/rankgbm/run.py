# -*- coding: utf-8 -*-
# ! /usr/bin/python

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
import time
import os


def init_out_dir(config, out_tag):
    # generate output tag
    # out_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    config.set('DIRECTORY', 'out_tag', str(out_tag))
    # generate output directory
    out_pt = config.get('DIRECTORY', 'out_pt')
    out_pt_exists = os.path.exists(out_pt)
    if out_pt_exists:
        LogUtil.log("ERROR", 'out path (%s) already exists ' % out_pt)
        raise Exception
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
            while len(features) <= fid:
                features.append(0.0)
            features[fid] = fvalue
        instances.append([rank, qid, features])
    file.close()
    LogUtil.log("INFO", "load rank file done. N(instances)=%d" % (len(instances)))
    return instances


def train(config, argv):
    fold_id = int(argv[0])
    out_tag = argv[1]
    init_out_dir(config, out_tag)
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

    train_file_name = '/mnt/disk2/xinyu/data/dataset/featwheel_vote_10_fe90ef2ad1a5f75899b6653ce822831b.fold%d_train.rank' % fold_id
    train_instances = load_rank_file(train_file_name)

    valid_file_name = '/mnt/disk2/xinyu/data/dataset/featwheel_vote_10_fe90ef2ad1a5f75899b6653ce822831b.fold%d_valid.rank' % fold_id
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


if __name__ == "__main__":
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)
