#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 7/25/17 9:49 PM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com

"""
v101 Change the parameters of v21:
    1. learning rate => 0.000087
    2. batch size => 128
    3. kernel count => 500 + 250
    4. Dense => 5000 
    5. Add one Dropout

v105 from v101:
    1. finetune from v101 519
    2. change logloss_with_2side_top_pn  5, 10
"""
