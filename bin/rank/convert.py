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
    inputs_pre = '/mnt/disk2/xinyu/data/tmp/'
    inputs = [
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.248.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v10.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.248.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v12.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v55.340.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/test.cnn-v55.360.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v1.284.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v1.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.280.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/test.lstm-v2.320.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.100.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.152.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v1.160.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.216.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/test.rcnn-v4.248.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.248.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v10.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.248.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v12.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.340.preds',
        'home/pangliang/niuox/zhihu-competition/preds/CNN/val.cnn-v55.360.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v1.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.280.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.300.preds',
        'home/pangliang/niuox/zhihu-competition/preds/LSTM/val.lstm-v2.320.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v1.152.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v1.160.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.200.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.216.preds',
        'home/pangliang/niuox/zhihu-competition/preds/RCNN/val.rcnn-v4.248.preds'
    ]

    outputs_pre = '/mnt/disk2/xinyu/data/dataset/'
    outputs = [
        'vote_fs_cnn_v10_200.online.csv',
        'vote_fs_cnn_v10_248.online.csv',
        'vote_fs_cnn_v10_300.online.csv',
        'vote_fs_cnn_v12_200.online.csv',
        'vote_fs_cnn_v12_248.online.csv',
        'vote_fs_cnn_v12_300.online.csv',
        'vote_fs_cnn_v55_340.online.csv',
        'vote_fs_cnn_v55_360.online.csv',
        'vote_fs_lstm_v1_284.online.csv',
        'vote_fs_lstm_v1_300.online.csv',
        'vote_fs_lstm_v2_280.online.csv',
        'vote_fs_lstm_v2_300.online.csv',
        'vote_fs_lstm_v2_320.online.csv',
        'vote_fs_rcnn_v1_100.online.csv',
        'vote_fs_rcnn_v1_152.online.csv',
        'vote_fs_rcnn_v1_160.online.csv',
        'vote_fs_rcnn_v4_200.online.csv',
        'vote_fs_rcnn_v4_216.online.csv',
        'vote_fs_rcnn_v4_248.online.csv',
        'vote_fs_cnn_v10_200.offline.csv',
        'vote_fs_cnn_v10_248.offline.csv',
        'vote_fs_cnn_v10_300.offline.csv',
        'vote_fs_cnn_v12_200.offline.csv',
        'vote_fs_cnn_v12_248.offline.csv',
        'vote_fs_cnn_v12_300.offline.csv',
        'vote_fs_cnn_v55_300.offline.csv',
        'vote_fs_cnn_v55_340.offline.csv',
        'vote_fs_cnn_v55_360.offline.csv',
        'vote_fs_lstm_v1_300.offline.csv',
        'vote_fs_lstm_v2_280.offline.csv',
        'vote_fs_lstm_v2_300.offline.csv',
        'vote_fs_lstm_v2_320.offline.csv',
        'vote_fs_rcnn_v1_152.offline.csv',
        'vote_fs_rcnn_v1_160.offline.csv',
        'vote_fs_rcnn_v4_200.offline.csv',
        'vote_fs_rcnn_v4_216.offline.csv',
        'vote_fs_rcnn_v4_248.offline.csv'
    ]

    assert len(inputs) == len(outputs)

    for i in range(len(inputs)):
        if 0 < inputs[i].count('test'):
            assert 0 < outputs[i].count('online')
        if 0 < inputs[i].count('val'):
            assert 0 < outputs[i].count('offline')
        print inputs[i]
        print outputs[i]
        inp = inputs_pre + inputs[i]
        out = outputs_pre + outputs[i]
        convert(inp, out)