#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/18 09:29
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
from data_utils import load_topic_info


def generate_word_share_features(config, argv):
    topic_info_fp = config.get('DIRECTORY', 'source_pt') + '/topic_info.txt'
    tid_list, father_list, tc_list, tw_list, dc_list, dw_list = load_topic_info(topic_info_fp)

    for tw in tw_list:
        print ' '.join(tw)


def main(argv):
    conf_fp = argv[1]
    cf = ConfigParser.ConfigParser()
    cf.read(conf_fp)
    func = argv[2]

    eval(func)(cf, argv[3:])


if __name__ == '__main__':
    main(sys.argv)
