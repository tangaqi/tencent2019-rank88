# -*- coding: utf-8 -*-
"""
@file:request_nums_position_part_test.py
@time:2019/6/9 13:50
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import time
'''
这里的request list是每个测试广告的请求集合，所以说取出来的对应的是广告id是唯一的值
也就是仍然是一个广告id对应一个行，但是会有不同的列，对应的是不同的广告位中的request nums
[2 1 4 3] 一共有4个广告位
'''
tt = time.time()
fea = pd.DataFrame()
request = pd.read_csv('../usingData/test/Request_list.csv')
log = pd.read_csv('../usingData/test/test_log.csv')
fea['ad_id'] = request['ad_id'].values
request_list = request['RequestList'].values
position1 = []
position2 = []
position3 = []
position4 = []

def f(x):
    xx = x.split(',')
    return xx[1]

for requests in request_list:
    temp = pd.DataFrame()
    re = requests.split('|')
    pos = list(map(f, re))
    temp['pos'] = pos
    p1 = temp[temp['pos'] == '1'].shape[0]
    p2 = temp[temp['pos'] == '2'].shape[0]
    p3 = temp[temp['pos'] == '3'].shape[0]
    p4 = temp[temp['pos'] == '4'].shape[0]
    position1.append(p1)
    position2.append(p2)
    position3.append(p3)
    position4.append(p4)


fea['p1_nums'] = position1
fea['p2_nums'] = position2
fea['p3_nums'] = position3
fea['p4_nums'] = position4

fea.to_csv('../usingData/feature/position_req_test.csv', index=False)
