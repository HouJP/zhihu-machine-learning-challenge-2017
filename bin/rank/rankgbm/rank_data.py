#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/11 22:35
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import hashlib
import ConfigParser
import sys
from ...utils import DataUtil, LogUtil
from ...text_cnn.data_helpers import load_labels_from_file
from ...evaluation import F_by_ids
from ...featwheel.feature import Feature
from ...featwheel.runner import Runner


def generate_offline(config, argv):
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
    will_save = ('True' == config.get('RANK', 'will_save'))
    offline_features = Feature.load_all(config.get('DIRECTORY', 'dataset_pt'),
                                        all_feature_names,
                                        'offline',
                                        will_save)

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

    # ========================= fold 0 =========================================

    fold_id = 0

    # generete indexs
    rank_train_indexs = rank_p1_indexs + rank_p2_indexs
    rank_valid_indexs = rank_p3_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    save_rank(config, train_labels, train_features, 'fold%d_train' % fold_id)
    save_rank(config, valid_labels, valid_features, 'fold%d_valid' % fold_id)

    # ========================= fold 1 =========================================

    fold_id = 1

    # generete indexs
    rank_train_indexs = rank_p2_indexs + rank_p3_indexs
    rank_valid_indexs = rank_p1_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    save_rank(config, train_labels, train_features, 'fold%d_train' % fold_id)
    save_rank(config, valid_labels, valid_features, 'fold%d_valid' % fold_id)

    # ========================= fold 2 =========================================

    fold_id = 2

    # generete indexs
    rank_train_indexs = rank_p3_indexs + rank_p1_indexs
    rank_valid_indexs = rank_p2_indexs

    # generate DMatrix
    train_features, train_labels, _ = Runner._generate_data(rank_train_indexs, offline_labels, offline_features, -1)
    valid_features, valid_labels, _ = Runner._generate_data(rank_valid_indexs, offline_labels, offline_features, -1)

    save_rank(config, train_labels, train_features, 'fold%d_train' % fold_id)
    save_rank(config, valid_labels, valid_features, 'fold%d_valid' % fold_id)


def save_rank(config, labels, features, data_name):
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')

    file_path = '%s/featwheel_vote_%d_%s.%s.rank' % (config.get('DIRECTORY', 'dataset_pt'),
                                                                     vote_k,
                                                                     vote_k_label_file_name,
                                                                     data_name)

    f = open(file_path, 'w')
    for lid in range(len(labels)):
        f.write('%d qid:%d %s\n' % (labels[lid], lid / vote_k, ' '.join(
            ['%d:%s' % (kv[0], kv[1]) for kv in enumerate(features[lid].toarray().tolist()[0])])))
    f.close()


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)