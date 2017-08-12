# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys

reload(sys)
sys.path.append("..")
sys.setdefaultencoding('utf-8')

import numpy as np  
import json
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn import linear_model
from ...utils import LogUtil
from rankevaluation import RankEvaluation
from ...evaluation import F_by_ids

reload(sys)
sys.setdefaultencoding('utf-8')


def self_define_f(preds, labels, vote_k):
    labels = zip(*[iter(list(labels))] * vote_k)
    preds = zip(*[iter(preds)] * vote_k)

    preds_ids = list()
    for i in range(len(preds)):
        preds_ids.append(
            [kv[0] for kv in sorted(enumerate(preds[i]), key=lambda x: x[1], reverse=True)])

    return F_by_ids(preds_ids, labels)


class RankGBM(object):

    def __init__(self, vote_k, n_round = 100, max_depth = 5, max_features = "auto", min_samples_leaf = 0.025, learn_rate = 0.2, silent = True):
        '''
        初始化模型参数
        '''
        LogUtil.log("INFO", "n_round=%d, max_depth=%d, max_features=%s, min_samples_leaf=%f, learn_rate=%f, silent=%d" 
            % (n_round, max_depth, max_features, min_samples_leaf, learn_rate, silent))
        # gradient boosting machine参数
        self.n_round = n_round
        # weak learner(决策树)参数
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.learn_rate = learn_rate
        # 数据信息
        self.vote_k = vote_k
        self.n_instances = 0
        self.qids = []
        self.n_qids = 0
        self.qid2did = {}
        self.coefficient_vec = {}
        self.weak_learners = []
        # 日志参数
        self.silent = silent
        return

    def fit(self, instances, watch_window):
        '''
        训练 rank gradient boosting machine 模型
        '''
        # 清空原先模型
        del self.weak_learners[:]
        # 获取数据信息
        self.n_instances = len(instances)
        self.qids = list(set([ instance[1] for instance in instances ]))
        self.n_qids = len(self.qids)
        self.qid2did = self.build_qid2did(instances)
        self.coefficient_vec = self.build_coefficient_vector(self.qid2did, instances)
        LogUtil.log("INFO", "n_instances=%d, n_qids=%d" % (self.n_instances, self.n_qids))

        # 生成特征矩阵
        Xs = np.array([ instances[i][2] for i in range (self.n_instances)])
        # 生成预测矩阵
        fs = [ 0.0 for i in range(self.n_instances) ]
        # 生成残差矩阵
        gs = [ 0.0 for i in range(self.n_instances) ]
        # 监视窗口生成特征矩阵
        watch_window_Xs = {}
        for dataset_name in watch_window:
            watch_window_Xs[dataset_name] = np.array( [ watch_window[dataset_name][i][2] for i in range(len(watch_window[dataset_name])) ])
        # 监视窗口生成预测矩阵
        watch_window_fs = {}
        for dataset_name in watch_window:
            watch_window_fs[dataset_name] = [ 0.0 for i in range(len(watch_window[dataset_name])) ]
        # 监视窗口生成映射：<qid, dids>
        watch_window_qid2did = {}
        for dataset_name in watch_window:
            watch_window_qid2did[dataset_name] = self.build_qid2did(watch_window[dataset_name])
        # 监视窗口生成标签向量
        watch_window_label = {}
        for dataset_name in watch_window:
            watch_window_label[dataset_name] = [ watch_window[dataset_name][i][0] for i in range(len(watch_window[dataset_name])) ]

        # Early Stop监视
        best_iter = 0
        best_em = 0.0
        best_vali_map = 0.0
        best_vali_ndcg10 = 0.0

        # 迭代拟合
        for iter in range(self.n_round):
            self.calculate_gradient_npos(instances, fs, gs)
            # 拟合单个weak learner
            neg_gradient = np.array([-1.0 * gs[did] for did in range(self.n_instances)])
            wl = DecisionTreeRegressor(max_depth = self.max_depth,  
                max_features = self.max_features, 
                min_samples_leaf = self.min_samples_leaf)
            wl.fit(Xs, neg_gradient)
            # 记录模型
            self.weak_learners.append(wl)
            # 利用单个模型预测
            ps = wl.predict(Xs)
            for did in range(self.n_instances):
                fs[did] += self.learn_rate * ps[did]
            LogUtil.log("INFO", "the %dth iteration done." % (iter + 1))
            # 预测监视窗口数据
            for dataset_name in watch_window:
                ww_ps = wl.predict(watch_window_Xs[dataset_name])
                for did in range(len(ww_ps)):
                    watch_window_fs[dataset_name][did] += self.learn_rate * ww_ps[did]

            vali_map = RankEvaluation.map(watch_window['vali'], watch_window_fs['vali'], watch_window_qid2did['vali'])
            vali_ndcg10 = RankEvaluation.ave_ndcg(watch_window['vali'], watch_window_fs['vali'], watch_window_qid2did['vali'], 10)
            valid_f = self_define_f(watch_window_fs['vali'], watch_window_label['vali'], self.vote_k)
            LogUtil.log("INFO", "vali\tMAP(%.4f)\tNDCG@10(%.4f)" % (vali_map, vali_ndcg10))
            # Early Stop逻辑
            if (vali_map + vali_ndcg10 > best_em):
                best_em = vali_map + vali_ndcg10
                best_iter = iter
                best_vali_map = vali_map
                best_vali_ndcg10 = vali_ndcg10
            elif (iter - best_iter >= 100):
                break
        self.n_round = best_iter + 1

        LogUtil.log("INFO", "rank gradient boosting machine done, n_round=%d, max_depth=%d, max_features=%s, min_samples_leaf=%f, learn_rate=%f" 
            % (self.n_round, self.max_depth, self.max_features, self.min_samples_leaf, self.learn_rate))
        LogUtil.log("INFO", "vali\tMAP(%.4f)\tNDCG@10(%.4f)" % (best_vali_map, best_vali_ndcg10))
        return

    def calculate_gradient_npos(self, instances, fs, gs):
        '''
        计算梯度，1 / |C| * ( sum(exp(-(fp-fj))) - sum(exp(-(fj - fq))) )
        '''
        for did in range(self.n_instances):
            exp_sum = 0.0
            rank = instances[did][0]
            qid = instances[did][1]
            dids = self.qid2did[qid]
            for did_pair in dids:
                rank_pair = instances[did_pair][0]
                if (rank_pair - rank > 1e-6):
                    exp_sum += math.pow(math.e, fs[did] - fs[did_pair])
                elif (rank - rank_pair > 1e-6):
                    exp_sum -= math.pow(math.e, fs[did_pair] - fs[did])
            gs[did] = self.coefficient_vec[qid] * exp_sum
        return

    def calculate_gradient_pos(self, instances, fs, gs):
        '''
        计算梯度，包含位置信息
        '''
        for did in range(self.n_instances):
            exp_sum = 0.0
            rank = instances[did][0]
            qid = instances[did][1]
            dids = self.qid2did[qid]

            dids_fss = {}
            for did_pair in dids:
                dids_fss[did_pair] = fs[did_pair]
            dids_fss_sorted = sorted(dids_fss.iteritems(), key=lambda d:d[1], reverse = True)
            dids_poss = {}
            for pos_id in range(len(dids_fss_sorted)):
                dids_poss[dids_fss_sorted[pos_id][0]] = pos_id + 1 

            for did_pair in dids:
                rank_pair = instances[did_pair][0]
                exp_diff = 0.0
                if (rank_pair - rank > 1e-6):
                    exp_diff = +1.0 * math.pow(math.e, fs[did] - fs[did_pair])
                elif (rank - rank_pair > 1e-6):
                    exp_diff = -1.0 * math.pow(math.e, fs[did_pair] - fs[did])
                max_pos = max(dids_poss[did_pair], dids_poss[did_pair])
                exp_sum += exp_diff / math.log(1 + max_pos, 2)
            gs[did] = 0.5 * self.coefficient_vec[qid] * exp_sum 
        return

    def build_qid2did(self, instances):
        '''
        建立从查询ID到文档ID集合的映射
        '''
        qid2did = {}
        for did in range(len(instances)):
            qid = instances[did][1]
            if qid not in qid2did:
                qid2did[qid] = []
            qid2did[qid].append(did)
        return qid2did

    def build_coefficient_vector(self, qid2did, instances):
        '''
        建立系数向量： 1 / | <di, dj> |, where rank_i > rank_j
        '''
        coefficient_vec = {}
        for qid in qid2did:
            if qid not in coefficient_vec:
                coefficient_vec[qid] = 0.0
            dids = qid2did[qid]
            n_ranks = {}
            for did in dids:
                rank = instances[did][0]
                if rank not in n_ranks:
                    n_ranks[rank] = 0
                n_ranks[rank] += 1
            sum = 0.0
            for rank in n_ranks:
                coefficient_vec[qid] += sum * n_ranks[rank]
                sum += n_ranks[rank]
            coefficient_vec[qid] = 1.0 / (coefficient_vec[qid] + 1.0)
        return coefficient_vec




    def predict(self, Xs):
        '''
        根据rank gradient boosting machine 模型进行预测
        '''
        fs = [ 0.0 for i in range(len(Xs)) ]
        for iter in range(self.n_round):
            # 利用单个模型进行预测
            ps = self.weak_learners[iter].predict(Xs)
            for did in range(len(Xs)):
                fs[did] += self.learn_rate * ps[did]
        LogUtil.log("INFO", "rank gradient boost predict done.")
        return fs



