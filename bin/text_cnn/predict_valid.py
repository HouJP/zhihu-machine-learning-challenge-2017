#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/28 10:42
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import sys, ConfigParser
from os.path import isfile, join
from os import listdir
import re


def extract_data(regex, content, index=1):
    r = 'nan'
    p = re.compile(regex)
    m = p.search(content)
    if m:
        r = m.group(index)
    return r


def predict_val(config):
    model_pt = config.get('DIRECTORY', 'model_pt')
    model_files = [f for f in listdir(model_pt) if isfile(join(model_pt, f))]
    part_id = [extract_data(r'text_cnn_(.*)\.', fn, 1) for fn in model_files]

    print part_id


if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)

    predict_val(config)