#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/30 23:54
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


def convert(inp, out):
    inp = open(inp, 'r')
    out = open(out, 'w')

    for line in inp:
        out.write(','.join([kv.split(':')[1] for kv in line.strip('\n').split(',')]) + '\n')

    out.close()
    inp.close()

if __name__ == '__main__':
    # inputs_pre = '/mnt/disk2/xinyu/data/tmp/'
    # inputs = [
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.248.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.248.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v55.340.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v55.360.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v1.284.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v1.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.280.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.320.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.100.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.152.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.160.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.216.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.248.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.248.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.248.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.340.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.360.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v1.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.280.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.300.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.320.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v1.152.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v1.160.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.200.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.216.preds',
    #     'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.248.preds'
    # ]
    #
    # outputs_pre = '/mnt/disk2/xinyu/data/dataset/'
    # outputs = [
    #     'vote_fs_cnn_v10_200.online.csv',
    #     'vote_fs_cnn_v10_248.online.csv',
    #     'vote_fs_cnn_v10_300.online.csv',
    #     'vote_fs_cnn_v12_200.online.csv',
    #     'vote_fs_cnn_v12_248.online.csv',
    #     'vote_fs_cnn_v12_300.online.csv',
    #     'vote_fs_cnn_v55_340.online.csv',
    #     'vote_fs_cnn_v55_360.online.csv',
    #     'vote_fs_lstm_v1_284.online.csv',
    #     'vote_fs_lstm_v1_300.online.csv',
    #     'vote_fs_lstm_v2_280.online.csv',
    #     'vote_fs_lstm_v2_300.online.csv',
    #     'vote_fs_lstm_v2_320.online.csv',
    #     'vote_fs_rcnn_v1_100.online.csv',
    #     'vote_fs_rcnn_v1_152.online.csv',
    #     'vote_fs_rcnn_v1_160.online.csv',
    #     'vote_fs_rcnn_v4_200.online.csv',
    #     'vote_fs_rcnn_v4_216.online.csv',
    #     'vote_fs_rcnn_v4_248.online.csv',
    #     'vote_fs_cnn_v10_200.offline.csv',
    #     'vote_fs_cnn_v10_248.offline.csv',
    #     'vote_fs_cnn_v10_300.offline.csv',
    #     'vote_fs_cnn_v12_200.offline.csv',
    #     'vote_fs_cnn_v12_248.offline.csv',
    #     'vote_fs_cnn_v12_300.offline.csv',
    #     'vote_fs_cnn_v55_300.offline.csv',
    #     'vote_fs_cnn_v55_340.offline.csv',
    #     'vote_fs_cnn_v55_360.offline.csv',
    #     'vote_fs_lstm_v1_300.offline.csv',
    #     'vote_fs_lstm_v2_280.offline.csv',
    #     'vote_fs_lstm_v2_300.offline.csv',
    #     'vote_fs_lstm_v2_320.offline.csv',
    #     'vote_fs_rcnn_v1_152.offline.csv',
    #     'vote_fs_rcnn_v1_160.offline.csv',
    #     'vote_fs_rcnn_v4_200.offline.csv',
    #     'vote_fs_rcnn_v4_216.offline.csv',
    #     'vote_fs_rcnn_v4_248.offline.csv'
    # ]

    # inputs_pre = '/home/xinyu/zhihu_preds/'
    # inputs = [
    #     'test.cnn-v52.208.preds',
    #     'test.cnn-v58.300.preds',
    #     'test.cnn-v58.340.preds',
    #     'test.cnn-v60.180.preds',
    #     'test.cnn-v60.240.preds',
    #     'test.rcnn-v4.216.preds',
    #     'test.rcnn-v4.264.preds',
    #     'test.rcnn-v4.300.preds',
    #     'val.cnn-v52.208.preds',
    #     'val.cnn-v58.300.preds',
    #     'val.cnn-v58.340.preds',
    #     'val.cnn-v60.180.preds',
    #     'val.cnn-v60.240.preds',
    #     'val.rcnn-v4.216.preds',
    #     'val.rcnn-v4.264.preds',
    #     'val.rcnn-v4.300.preds'
    # ]
    #
    # outputs_pre = '/mnt/disk2/xinyu/data/dataset/'
    # outputs = [
    #     'vote_fs_cnn_v52_208.online.csv',
    #     'vote_fs_cnn_v58_300.online.csv',
    #     'vote_fs_cnn_v58_340.online.csv',
    #     'vote_fs_cnn_v60_180.online.csv',
    #     'vote_fs_cnn_v60_240.online.csv',
    #     'vote_fs_rcnn_v4_216.online.csv',
    #     'vote_fs_rcnn_v4_264.online.csv',
    #     'vote_fs_rcnn_v4_300.online.csv',
    #     'vote_fs_cnn_v52_208.offline.csv',
    #     'vote_fs_cnn_v58_300.offline.csv',
    #     'vote_fs_cnn_v58_340.offline.csv',
    #     'vote_fs_cnn_v60_180.offline.csv',
    #     'vote_fs_cnn_v60_240.offline.csv',
    #     'vote_fs_rcnn_v4_216.offline.csv',
    #     'vote_fs_rcnn_v4_264.offline.csv',
    #     'vote_fs_rcnn_v4_300.offline.csv'
    # ]

    inputs_outputs = {
        '/home/xinyu/zhihu_preds/pack_niu/test.cnn-v61.300.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v61_300.online.csv',
        '/home/xinyu/zhihu_preds/pack_niu/test.cnn-v61.328.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v61_328.online.csv',
        '/home/xinyu/zhihu_preds/pack_niu/test.lstm-v1.244.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_lstm_v1_244.online.csv',
        '/home/xinyu/zhihu_preds/pack_niu/test.lstm-v1.284.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_lstm_v1_284.online.csv',
        '/home/xinyu/zhihu_preds/pack_niu/val.cnn-v61.300.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v61_300.offline.csv',
        '/home/xinyu/zhihu_preds/pack_niu/val.cnn-v61.328.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v61_328.offline.csv',
        '/home/xinyu/zhihu_preds/pack_niu/val.lstm-v1.244.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_lstm_v1_244.offline.csv',
        '/home/xinyu/zhihu_preds/pack_niu/val.lstm-v1.284.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_lstm_v1_284.offline.csv',
        '/home/xinyu/zhihu_preds/pack1/test.cnn-v64.272.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v64_272.online.csv',
        '/home/xinyu/zhihu_preds/pack1/test.cnn-v64.304.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v64_304.online.csv',
        '/home/xinyu/zhihu_preds/pack1/test.cnn-v65.304.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v65_304.online.csv',
        '/home/xinyu/zhihu_preds/pack1/test.cnn-v65.336.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v65_336.online.csv',
        '/home/xinyu/zhihu_preds/pack1/val.cnn-v64.272.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v64_272.offline.csv',
        '/home/xinyu/zhihu_preds/pack1/val.cnn-v64.304.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v64_304.offline.csv',
        '/home/xinyu/zhihu_preds/pack1/val.cnn-v65.304.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v65_304.offline.csv',
        '/home/xinyu/zhihu_preds/pack1/val.cnn-v65.336.preds':
            '/mnt/disk2/xinyu/data/dataset/vote_fs_cnn_v65_336.offline.csv',
    }

    # assert len(inputs) == len(outputs)

    for kv in inputs_outputs:
        inputs_i = kv[0]
        outputs_i = kv[1]
        if 0 < inputs_i.count('test'):
            assert 0 < outputs_i.count('online')
        if 0 < inputs_i.count('val'):
            assert 0 < outputs_i.count('offline')
        print inputs_i
        print outputs_i
        convert(inputs_i, outputs_i)