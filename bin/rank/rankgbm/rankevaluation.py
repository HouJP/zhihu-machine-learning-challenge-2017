# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys
reload(sys)
sys.path.append("..")
sys.setdefaultencoding('utf-8')

import math

from ...utils import LogUtil

class RankEvaluation(object):

    rank_max = 20
    gain = []
    for rank in range(rank_max):
        gain.append(math.pow(2.0, rank) - 1.0)

    pos_max = 2000
    discount_factor_rev = [1.0]
    for pos in range(1, pos_max):
        discount_factor_rev.append(math.log(1.0 + pos, 2))

    def __init__(self):
        

        pass

    @staticmethod
    def map(instances, ps, pid2did):
        '''
        计算`mean average precision`
        '''
        map_value = 0.0
        for pid in pid2did:
            dids = pid2did[pid]
            kv = {}
            for did in dids:
                kv[did] = ps[did]
            kv_sorted = sorted(kv.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            n_relevant = 0.0
            p_value = 0.0
            for i in range(len(kv_sorted)):
                did = kv_sorted[i][0]
                if (instances[did][0] > 1e-6):
                    n_relevant += 1.0
                    p_value += n_relevant / (i + 1)
            map_value += p_value / n_relevant if n_relevant > 1e-6 else 0.0
        map_value /= len(pid2did)
        return map_value

    @staticmethod
    def ndcg(p_r, pos):
        '''
        计算`NDCG(Normalized Discounted Cumulative Gain)`，针对一个query的预测结果
        '''
        dcg = 0.0
        dcg_max = 0.0
        p_r_sorted = sorted(p_r, key = lambda x: (-x[0], x[1]))
        r_sorted = sorted([ e[1] for e in p_r ], reverse = True)
        if (pos > len(p_r)):
            return 0.0
        for i in range(pos):
            dcg += RankEvaluation.gain[p_r_sorted[i][1]] / RankEvaluation.discount_factor_rev[i]
            dcg_max += RankEvaluation.gain[r_sorted[i]] / RankEvaluation.discount_factor_rev[i]
            # print "%d,(%d, %d), %f, %f" % (i, p_r_sorted[i][0], p_r_sorted[i][1], dcg, dcg_max)
        return 0.0 if (dcg_max < 1e-6) else (dcg / dcg_max)

    @staticmethod
    def ave_ndcg(instances, ps, pid2did, pos = sys.maxint):
        '''
        计算`NDCG(Normalized Discounted Cumulative Gain)`，针对所有query的预测结果求平均
        '''
        ave_ndcg = 0.0
        for pid in pid2did:
            dids = pid2did[pid]
            p_r = [ (ps[did], instances[did][0]) for did in dids ]
            ave_ndcg += RankEvaluation.ndcg(p_r, pos)
        return ave_ndcg / len(pid2did)


if __name__ == "__main__":
    p_r = [(0.446180, 0),(-0.161544, 0),(0.340309, 0),(0.159737, 1),(0.078041,0),(0.130209, 0),(-0.284462, 0),(-0.233077, 0)]
    pos = 10
    print RankEvaluation.ndcg(p_r, pos)

