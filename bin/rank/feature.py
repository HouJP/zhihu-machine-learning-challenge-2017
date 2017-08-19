#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 14:59
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys
import ConfigParser
import hashlib
import itertools
import os
from ..utils import DataUtil, LogUtil
from os.path import isfile
from ..featwheel.feature import Feature


def find_model_feature_file(model_name, data_name):
    RootDir = ['/mnt/disk2/xinyu/data/dataset/',
               '/home/xinyu/zhihu_preds/from_124/',
               '/mnt/disk3/xinyu/from_124/']
    RootDir.append('/mnt/disk2/xinyu/niuox_data/RCNN/')

    model_name = model_name.strip().strip('\n')
    FileTemp = ''
    if len(model_name) != 0 and model_name[0] != '#':
        for rd in RootDir:
            if os.path.isfile('%s/%s.%s.csv' % (rd, model_name, data_name)):
                FileTemp = '%s/%s.%s.csv' % (rd, model_name, data_name)
                break
            elif os.path.isfile('%s/%s.%s.preds' % (rd, model_name, data_name)):
                FileTemp = '%s/%s.%s.preds' % (rd, model_name, data_name)
                break
    if '' == FileTemp:
        LogUtil.log('INFO', 'can\'t find %s' % model_name)

    return FileTemp, model_name


def generate_featwheel_feature_from_model(config, argv):
    data_name = argv[0]

    # load vote_k ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name)
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    # load model features
    feature_raw_names = config.get('RANK', 'model_features').split()
    feature_names = list()
    feature_fps = list()
    for feature_raw_name in feature_raw_names:
        fp, fn = find_model_feature_file(feature_raw_name, data_name)
        if 0 < len(fp):
            feature_names.append(fn)
            feature_fps.append(fp)
    for fid, feature_name in enumerate(feature_names):
        LogUtil.log('INFO', 'model_feature=%s' % feature_name)

        featwheel_feature_file_path = '%s/featwheel_vote_%d_%s_%s.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'),
                                                                              vote_k,
                                                                              vote_k_label_file_name,
                                                                              feature_name,
                                                                              data_name)
        LogUtil.log('INFO', 'featwheel_feature_file_path=%s' % featwheel_feature_file_path)
        has_featwheel_features = isfile('%s' % featwheel_feature_file_path)
        if has_featwheel_features:
            LogUtil.log('INFO', 'has featwheel features, JUMP')
            continue

        features = DataUtil.load_matrix(feature_fps[fid], 'float')
        assert len(vote_k_label) == len(features)

        featwheel_feature_file = open(featwheel_feature_file_path, 'w')
        featwheel_feature_file.write('%d %d\n' % (len(features) * vote_k, 1))

        for line_id in range(len(features)):
            for lid in vote_k_label[line_id]:
                Feature.save_feature([features[line_id][lid]], featwheel_feature_file)

        featwheel_feature_file.close()


def generate_featwheel_feature_from_instance(config, argv):
    data_name = argv[0]

    # load vote_k ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name)
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    # load instance features
    feature_names = config.get('RANK', 'instance_features').split()
    for feature_name in feature_names:
        LogUtil.log('INFO', 'instance_feature=%s' % feature_name)

        featwheel_feature_file_path = '%s/featwheel_vote_%d_%s_%s.%s.smat' % (
            config.get('DIRECTORY', 'dataset_pt'),
            vote_k,
            vote_k_label_file_name,
            feature_name,
            data_name)
        LogUtil.log('INFO', 'featwheel_feature_file_path=%s' % featwheel_feature_file_path)
        has_featwheel_features = isfile('%s' % featwheel_feature_file_path)
        if has_featwheel_features:
            LogUtil.log('INFO', 'has featwheel features, JUMP')
            continue

        features = Feature.load_smat('%s/%s.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, data_name))
        assert len(vote_k_label) == (features.shape)[0]

        indexs = list(itertools.chain(*[[i] * vote_k for i in range((features.shape)[0])]))
        featwheel_features = Feature.sample_row(features, indexs)

        Feature.save_smat(featwheel_features, featwheel_feature_file_path)


def generate_featwheel_feature_from_topic(config, argv):
    data_name = argv[0]

    # load vote_k ids
    index_pt = config.get('DIRECTORY', 'index_pt')
    vote_feature_names = config.get('RANK', 'vote_features').split()
    vote_k_label_file_name = hashlib.md5('|'.join(vote_feature_names)).hexdigest()
    vote_k = config.getint('RANK', 'vote_k')
    vote_k_label_file_path = '%s/vote_%d_label_%s.%s.index' % (index_pt, vote_k, vote_k_label_file_name, data_name)
    vote_k_label = DataUtil.load_matrix(vote_k_label_file_path, 'int')

    # load topic features
    feature_names = config.get('RANK', 'topic_features').split()
    for feature_name in feature_names:
        LogUtil.log('INFO', 'topic_feature=%s' % feature_name)

        featwheel_feature_file_path = '%s/featwheel_vote_%d_%s_%s.%s.smat' % (
            config.get('DIRECTORY', 'dataset_pt'),
            vote_k,
            vote_k_label_file_name,
            feature_name,
            data_name)
        LogUtil.log('INFO', 'featwheel_feature_file_path=%s' % featwheel_feature_file_path)
        has_featwheel_features = isfile('%s' % featwheel_feature_file_path)
        if has_featwheel_features:
            LogUtil.log('INFO', 'has featwheel features, JUMP')
            continue

        features = Feature.load_smat('%s/%s.%s.smat' % (config.get('DIRECTORY', 'dataset_pt'), feature_name, 'all'))

        indexs = list(itertools.chain(*vote_k_label))
        featwheel_features = Feature.sample_row(features, indexs)

        Feature.save_smat(featwheel_features, featwheel_feature_file_path)


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)