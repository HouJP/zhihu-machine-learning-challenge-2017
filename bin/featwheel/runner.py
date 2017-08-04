#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/22 00:27
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import os
import random
import time
import hashlib
import re

from ..utils import LogUtil, DataUtil
from feature import Feature
from model import Model
from ..text_cnn.data_helpers import load_raw_line_from_file


class Runner(object):
    def __init__(self, config_fp):
        # load configuration file
        if 'str' == type(config_fp):
            self.config = ConfigParser.ConfigParser()
            self.config.read(config_fp)
        else:
            self.config = config_fp
        self.__init_out_dir()

    @staticmethod
    def __generate_data(indexs, labels, features, positive_rate):
        """
        generate data set according to the `indexs` and `positive_rate`
        :param indexs: indexs which will select data from raw data set
        :param labels: all labels of raw data set
        :param features: feature matrix
        :param positive_rate: positive_rate in data set
        :return: feature matrix, labels, balanced indexs
        """
        # balance the data set
        balanced_indexs = Feature.balance_index(indexs, labels, positive_rate)
        # sample labels
        labels = [labels[index] for index in balanced_indexs]
        # sample features
        features = Feature.sample_row(features, balanced_indexs)

        return features, labels, balanced_indexs

    def __init_out_dir(self):
        # generate output tag
        self.out_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.config.set('DIRECTORY', 'out_tag', str(self.out_tag))
        # generate output directory
        out_pt = self.config.get('DIRECTORY', 'out_pt')
        out_pt_exists = os.path.exists(out_pt)
        if out_pt_exists:
            LogUtil.log("ERROR", 'out path (%s) already exists ' % out_pt)
            return
        else:
            os.mkdir(out_pt)
            os.mkdir(self.config.get('DIRECTORY', 'pred_pt'))
            os.mkdir(self.config.get('DIRECTORY', 'model_pt'))
            os.mkdir(self.config.get('DIRECTORY', 'conf_pt'))
            LogUtil.log('INFO', 'out path (%s) created ' % out_pt)
        # save config
        self.config.write(open(self.config.get('DIRECTORY', 'conf_pt') + 'featwheel.conf', 'w'))


class SingleExec(Runner):
    def __init__(self, config_fp):
        Runner.__init__(self, config_fp)
        self.se_num, self.se_tag = self.__load_parameters()

    def __load_parameters(self):
        se_num = self.config.getint('SINGLE_EXEC', 'se_num')
        se_tag = self.config.get('SINGLE_EXEC', 'se_tag')
        return se_num, se_tag

    def __generate_index(self, row_num, vote_k, rank_k):
        train_indexs = list()
        valid_indexs = list()
        train_instance_indexs = list()
        valid_instance_indexs = list()
        for i in range(row_num / vote_k):
            part_id = random.random() * self.se_num
            ids = [i * vote_k + j for j in range(rank_k)]
            if part_id < self.se_num - 1:
                train_indexs.extend(ids)
                train_instance_indexs.append(i)
            else:
                valid_indexs.extend(ids)
                valid_instance_indexs.append(i)
        index_pt = self.config.get('DIRECTORY', 'index_pt')
        train_fp = '%s/se_tag%s_train.%s.index' % (index_pt,
                                                   self.se_tag,
                                                   'offline')
        DataUtil.save_vector(train_fp, train_indexs, 'w')
        valid_fp = '%s/se_tag%s_valid.%s.index' % (index_pt,
                                                   self.se_tag,
                                                   'offline')
        DataUtil.save_vector(valid_fp, valid_indexs, 'w')

        train_instance_fp = '%s/se_tag%s_train_instance.%s.index' % (index_pt,
                                                                     self.se_tag,
                                                                     'offline')
        DataUtil.save_vector(train_instance_fp, train_instance_indexs, 'w')
        valid_instance_fp = '%s/se_tag%s_valid_instance.%s.index' % (index_pt,
                                                                     self.se_tag,
                                                                     'offline')
        DataUtil.save_vector(valid_instance_fp, valid_instance_indexs, 'w')

    def run_offline(self):
        # load parameters
        vote_k = self.config.getint('RANK', 'vote_k')
        rank_k = self.config.getint('RANK', 'rank_k')

        # load feture names
        feature_names = self.config.get('RANK', 'model_features').split() + \
                        self.config.get('RANK', 'instance_features').split() + \
                        self.config.get('RANK', 'topic_features').split()
        feature_names = ['featwheel_vote_%d_%s' % (vote_k, fn) for fn in feature_names]

        # load feature matrix
        offline_features = Feature.load_all(self.config.get('DIRECTORY', 'dataset_pt'),
                                            feature_names,
                                            'offline',
                                            False)
        # load labels
        offline_labels_file_path = '%s/featwheel_vote_%d.%s.label' % (self.config.get('DIRECTORY', 'label_pt'),
                                                                      vote_k,
                                                                      'offline')
        offline_labels = DataUtil.load_vector(offline_labels_file_path,
                                              'int')
        # generate index file
        if '' == self.se_tag:
            LogUtil.log('INFO', 'se_tag=%s, generate index' % self.out_tag)
            self.se_tag = self.out_tag
            self.__generate_index(offline_features.shape[0], vote_k, rank_k)
        index_pt = self.config.get('DIRECTORY', 'index_pt')
        # generate training data set
        offline_train_pos_rate = -1
        offline_train_indexs_fp = '%s/se_tag%s_train.%s.index' % (index_pt,
                                                                  self.se_tag,
                                                                  'offline')
        offline_train_indexs = DataUtil.load_vector(offline_train_indexs_fp, 'int')
        offline_train_features, offline_train_labels, offline_train_balanced_indexs = \
            self.__generate_data(offline_train_indexs,
                                       offline_labels,
                                       offline_features,
                                       offline_train_pos_rate)
        LogUtil.log('INFO', 'offline train data generation done')

        # generate validation data set
        offline_valid_pos_rate = -1
        offline_valid_indexs_fp = '%s/se_tag%s_valid.%s.index' % (index_pt,
                                                                  self.se_tag,
                                                                  'offline')
        offline_valid_indexs = DataUtil.load_vector(offline_valid_indexs_fp, 'int')
        offline_valid_features, offline_valid_labels, offline_valid_balanced_indexs = \
            self.__generate_data(offline_valid_indexs,
                                       offline_labels,
                                       offline_features,
                                       offline_valid_pos_rate)
        LogUtil.log('INFO', 'offline valid data generation done')

        model = Model.new('XGB', self.config)
        model_fp = self.config.get('DIRECTORY', 'model_pt') + '/se.%s.model' % 'xgb'
        offline_valid_preds = model.fit(offline_train_features,
                                        offline_train_labels,
                                        offline_valid_features,
                                        offline_valid_labels)
        model.save(model_fp)

        # load vote_k ids
        index_pt = self.config.get('DIRECTORY', 'index_pt')
        vote_feature_names = self.config.get('RANK', 'vote_features').split()
        vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
        vote_k = self.config.getint('RANK', 'vote_k')
        vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, 'offline')
        vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')
        valid_instance_fp = '%s/se_tag%s_valid_instance.%s.index' % (index_pt,
                                                                     self.se_tag,
                                                                     'offline')
        valid_labels_lines = load_raw_line_from_file(self.config, vote_k_label_file_path,
                                                     DataUtil.load_vector(valid_instance_fp, 'int'))

        return offline_valid_preds

    def run_online(self):
        # load feature matrix
        online_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                           self.config.get('FEATURE', 'feature_selected').split(),
                                           self.config.get('MODEL', 'online_rawset_name'),
                                           self.config.get('FEATURE', 'will_save'))
        model = Model.new(self.config.get('MODEL', 'model_name'), self.config)
        model_fp = self.config.get('DIRECTORY', 'model_pt') + '/se.%s.model' % self.config.get('MODEL', 'model_name')
        model.load(model_fp)
        online_preds = model.predict(online_features)
        online_preds_fp = '%s/se_online.%s.pred' % (self.config.get('DIRECTORY', 'pred_pt'),
                                                    self.config.get('MODEL', 'online_test_rawset_name'))
        DataUtil.save_vector(online_preds_fp, online_preds, 'w')


class CrossValidation(Runner):
    def __init__(self, config_fp):
        Runner.__init__(self, config_fp)
        self.cv_num, self.cv_tag = self.__load_parameters()

    def __load_parameters(self):
        cv_num = self.config.get('CROSS_VALIDATION', 'cv_num')
        cv_tag = self.config.get('CROSS_VALIDATION', 'cv_tag')
        return cv_num, cv_tag

    def __generate_index(self, row_num):
        index_all = [list()] * self.cv_num
        for i in range(row_num):
            index_all[int(random.random() * self.cv_num)].append(i)
        for i in range(self.cv_num):
            LogUtil.log('INFO', 'generate cv index, size(part%d)=%d' % (i, len(index_all[i])))

        index_pt = self.config.get('DEFAULT', 'index_pt')
        for i in range(self.cv_num):
            fold_id = i
            # train
            fp = '%s/cv_tag%s_n%d_f%d_train.%s.index' % (index_pt,
                                                         self.cv_tag,
                                                         self.cv_num,
                                                         fold_id,
                                                         self.config.get('MODEL', 'offline_rawset_name'))
            DataUtil.save_vector(fp, list(), 'w')
            for j in range(self.cv_num - 2):
                part_id = (i + j) % self.cv_num
                DataUtil.save_vector(fp, index_all[part_id], 'a')
            # valid
            fp = '%s/cv_tag%s_n%d_f%d_valid.%s.index' % (index_pt,
                                                         self.cv_tag,
                                                         self.cv_num,
                                                         fold_id,
                                                         self.config.get('MODEL', 'offline_rawset_name'))
            part_id = (fold_id + self.cv_num - 2) % self.cv_num
            DataUtil.save_vector(fp, index_all[part_id], 'w')
            # test
            fp = '%s/cv_tag%s_n%d_f%d_test.%s.index' % (index_pt,
                                                        self.cv_tag,
                                                        self.cv_num,
                                                        fold_id,
                                                        self.config.get('MODEL', 'offline_rawset_name'))
            part_id = (fold_id + self.cv_num - 1) % self.cv_num
            DataUtil.save_vector(fp, index_all[part_id], 'w')

    def run_offline(self):
        LogUtil.log('INFO', 'cv_tag(%s)' % self.cv_tag)
        # load feature matrix
        offline_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                            self.config.get('FEATURE', 'feature_selected').split(),
                                            self.config.get('MODEL', 'offline_rawset_name'),
                                            self.config.get('FEATURE', 'will_save'))
        # load labels
        offline_labels = DataUtil.load_vector('%s/%s.label' % (self.config.get('DIRECTORY', 'label_pt'),
                                                               self.config.get('MODEL', 'offline_rawset_name')),
                                              True)
        # generate index file
        if '' == self.cv_tag:
            self.cv_tag = self.out_tag
            self.__generate_index(offline_features.shape[0])
        # cross validation
        offline_valid_preds_all = [0.] * offline_features.shape[0]
        offline_test_preds_all = [0.] * offline_features.shape[0]
        for fold_id in range(self.cv_num):
            LogUtil.log('INFO', 'cross validation fold_id(%d) begin' % fold_id)

            # generate training data set
            offline_train_pos_rate = float(self.config.get('MODEL', 'train_pos_rate'))
            offline_train_indexs_fp = '%s/cv_tag%s_n%d_f%d_train.%s.index' % (self.config.get('DIRECTORY', 'index_pt'),
                                                                              self.cv_tag,
                                                                              self.cv_num,
                                                                              fold_id,
                                                                              self.config.get('MODEL',
                                                                                              'offline_rawset_name'))
            offline_train_indexs = DataUtil.load_vector(offline_train_indexs_fp, 'int')
            offline_train_features, offline_train_labels, offline_train_balanced_indexs = \
                CrossValidation.__generate_data(offline_train_indexs,
                                                offline_labels,
                                                offline_features,
                                                offline_train_pos_rate)
            LogUtil.log('INFO', 'offline train data generation done')

            # generate validation data set
            offline_valid_pos_rate = float(self.config.get('MODEL', 'valid_pos_rate'))
            offline_valid_indexs_fp = '%s/cv_tag%s_n%d_f%d_valid.%s.index' % (self.config.get('DIRECTORY', 'index_pt'),
                                                                              self.cv_tag,
                                                                              self.cv_num,
                                                                              fold_id,
                                                                              self.config.get('MODEL',
                                                                                              'offline_rawset_name'))
            offline_valid_indexs = DataUtil.load_vector(offline_valid_indexs_fp, 'int')
            offline_valid_features, offline_valid_labels, offline_valid_balanced_indexs = \
                CrossValidation.__generate_data(offline_valid_indexs,
                                                offline_labels,
                                                offline_features,
                                                offline_valid_pos_rate)
            LogUtil.log('INFO', 'offline valid data generation done')

            # generate test data set
            offline_test_pos_rate = float(self.config.get('MODEL', 'test_pos_rate'))
            offline_test_indexs_fp = '%s/cv_tag%s_n%d_f%d_test.%s.index' % (self.config.get('DIRECTORY', 'index_pt'),
                                                                            self.cv_tag,
                                                                            self.cv_num,
                                                                            fold_id,
                                                                            self.config.get('MODEL',
                                                                                            'offline_rawset_name'))
            offline_test_indexs = DataUtil.load_vector(offline_test_indexs_fp, 'int')
            offline_test_features, offline_test_labels, offline_test_balanced_indexs = \
                CrossValidation.__generate_data(offline_test_indexs,
                                                offline_labels,
                                                offline_features,
                                                offline_test_pos_rate)
            LogUtil.log('INFO', 'offline test data generation done')

            model = Model.new(self.config.get('MODEL', 'model_name'), self.config)
            model_fp = self.config.get('DIRECTORY', 'model_pt') + '/cv_n%d_f%d.%s.model' % \
                                                                  (self.cv_num,
                                                                   fold_id,
                                                                   self.config.get('MODEL', 'model_name'))
            model.save(model_fp)
            offline_train_preds, offline_valid_preds, offline_test_preds = model.fit(offline_train_features,
                                                                                     offline_train_labels,
                                                                                     offline_valid_features,
                                                                                     offline_valid_labels,
                                                                                     offline_test_features,
                                                                                     offline_test_labels)
            offline_train_score = Evaluator.evaluate(self.config.get('MODEL', 'evaluator_name'),
                                                     offline_train_labels,
                                                     offline_train_preds)
            offline_valid_score = Evaluator.evaluate(self.config.get('MODEL', 'evaluator_name'),
                                                     offline_valid_labels,
                                                     offline_valid_preds)
            offline_test_score = Evaluator.evaluate(self.config.get('MODEL', 'evaluator_name'),
                                                    offline_test_labels,
                                                    offline_test_preds)
            score_fp = '%s/%s.score' % (self.config.get('DIRECTORY', 'score_pt'), 'cv')
            score_file = open(score_fp, 'a')
            score_file.write('fold:%d\ttrain:%s\tvalid:%s\ttest:%s\n' % (fold_id,
                                                                         offline_train_score,
                                                                         offline_valid_score,
                                                                         offline_test_score))
            score_file.close()
            # merge prediction results
            for index in range(len(offline_valid_balanced_indexs)):
                offline_valid_preds_all[offline_valid_balanced_indexs[index]] = offline_valid_preds[index]
            for index in range(len(offline_test_balanced_indexs)):
                offline_test_preds_all[offline_test_balanced_indexs[index]] = offline_test_preds[index]
            LogUtil.log('INFO', 'cross test fold_id(%d) done' % fold_id)
        # save prediction results
        offline_valid_preds_all_fp = '%s/cv_n%d_valid.%s.pred' % (self.config.get('DIRECTORY', 'pred_pt'),
                                                                  self.cv_num,
                                                                  self.config.get('MODEL', 'offline_rawset_name'))
        DataUtil.save_vector(offline_valid_preds_all_fp, offline_valid_preds_all, 'w')
        offline_test_preds_all_fp = '%s/cv_n%d_test.%s.pred' % (self.config.get('DIRECTORY', 'pred_pt'),
                                                                self.cv_num,
                                                                self.config.get('MODEL', 'offline_rawset_name'))
        DataUtil.save_vector(offline_test_preds_all_fp, offline_test_preds_all, 'w')
        # evaluate
        offline_valid_score = Evaluator.evaluate(self.config.get('MODEL', 'evaluator_name'),
                                                 offline_labels,
                                                 offline_valid_preds_all)
        offline_test_score = Evaluator.evaluate(self.config.get('MODEL', 'evaluator_name'),
                                                offline_labels,
                                                offline_test_preds_all)
        score_fp = '%s/%s.score' % (self.config.get('DIRECTORY', 'score_pt'), 'cv')
        score_file = open(score_fp, 'a')
        score_file.write('cross_validation\tvalid:%s\ttest:%s\n' % (offline_valid_score, offline_test_score))
        score_file.close()

    def run_online(self):
        # load feature matrix
        online_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                           self.config.get('FEATURE', 'feature_selected').split(),
                                           self.config.get('MODEL', 'online_rawset_name'),
                                           self.config.get('FEATURE', 'will_save'))
        for fold_id in range(self.cv_num):
            model = Model.new(self.config.get('MODEL', 'model_name'), self.config)
            model_fp = self.config.get('DIRECTORY', 'model_pt') + '/cv_n%d_f%d.%s.model' % \
                                                                  (self.cv_num,
                                                                   fold_id,
                                                                   self.config.get('MODEL', 'model_name'))
            model.load(model_fp)
            online_preds = model.predict(online_features)
            online_preds_fp = '%s/cv_n%d_f%d_online.%s.pred' % (self.config.get('DIRECTORY', 'pred_pt'),
                                                                self.cv_num,
                                                                fold_id,
                                                                self.config.get('MODEL', 'online_test_rawset_name'))
            DataUtil.save_vector(online_preds_fp, online_preds, 'w')
