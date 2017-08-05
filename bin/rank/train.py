#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 8/4/17 5:08 PM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com


import ConfigParser
import sys
from ..featwheel.runner import SingleExec

def single_execute(config, argv):
    single_exe = SingleExec(config)
    valid_preds = single_exe.run_offline()



if __name__ == '__main__':
    config_fp = sys.argv[1]
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    func = sys.argv[2]
    argv = sys.argv[3:]

    eval(func)(config, argv)