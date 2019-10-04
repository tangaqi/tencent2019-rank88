# -*- coding: utf-8 -*-
"""
@file:test_feature2.py
@time:2019/6/5 14:49
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import time
tt = time.time()
fea = pd.DataFrame()
test_bid = pd.read_csv('../usingData/test/test_bid.csv')
request = pd.read_csv('../usingData/test/Request_list.csv')
log = pd.read_csv('../usingData/test/test_log.csv')
log = log.fillna(-100)
print(request.columns)
print(time.localtime(tt))
'''
竞争队列中的pctr、ecpm、bid等信息的均值，用的是日志文件中竞争队列里面当天的pctr等信息
找出对应的人群的，每一个广告都有一个dataframe，然后将那个log文件和它merge起来
也就是这个广告对应的竞争队列，取出竞争队列中的前两个，取均值，应该先输出一下竞争队列的那些值是不是已经排好序的
'''
request_list = request['RequestList'].values
test_aid = request['ad_id'].values
print(len(request_list))
print(len(test_aid))
log_compete = log['competeAd'].values
log_reqid = log['RequestId'].values
log_posiId = log['PositionId'].values
request = []
for i, k in enumerate(log_reqid):
    s = str(log_reqid[i]) + ',' + str(log_posiId[i])
    request.append(s)
log_new = pd.DataFrame()
log_new['compete'] = log_compete
log_new['request'] = np.array(request)
uid_len = []
k = 0

test_com_pctr = []
test_quality_ecpm = []
test_total_ecpm = []
test_com_bid = []
test_new_aid = []

# 找出每个aid对应的request集合，然后作为一个dataframe，和上面的进行merge，找出其对应竞争队列
for request_l in request_list:
    tt = time.time()
    request_i = request_l.split('|')
    request_new = pd.DataFrame()
    request_new['request'] = request_i
    request_new = pd.merge(request_new, log_new, on='request', how='left')
    # print(request_new)
    # 这个是竞争队列的集合，每一个都要取前两个，且是没有被屏蔽的
    competeAd = request_new['compete'].values
    def f1(x):
        xx = x.split(',')
        t = xx[-1]
        return t
    com_pctr = []
    quality_ecpm = []
    total_ecpm = []
    com_bid = []
    for list_i in competeAd:
        if list_i == -100:
            continue
        if pd.isna(list_i):
            continue
        ads = list_i.split(';')
        print(type(ads))
        index = list(map(f1, ads))
        # list(map(deal, ads))
        # 找出两个可以用的index
        index1 = 0
        index2 = 0
        for i, k in enumerate(index):
            if (k != 1) & (index1 == 0):
                index1 = i
                continue
            if (k != 1) & (index2 == 0):
                index2 = i
                continue
            if (index1 != 0) & (index2 != 0):
                break
        print(index1)
        print(index2)
        aid1 = ads[index1].split(',')
        aid2 = ads[index2].split(',')
        temp_bid1 = (float(aid2[1]) + float(aid1[1])) / 2
        temp_pctr1 = (float(aid2[2]) + float(aid1[2])) / 2
        temp_qu_ecpm1 = (float(aid2[3]) + float(aid1[3])) / 2
        temp_to_ecpm1 = (float(aid2[4]) + float(aid1[4])) / 2

        com_bid.append(temp_bid1)
        com_pctr.append(temp_pctr1)
        quality_ecpm.append(temp_qu_ecpm1)
        total_ecpm.append(temp_to_ecpm1)
    # 直接将取出来的数组，取均值
    test_com_pctr.append(np.mean(com_pctr))
    test_quality_ecpm.append(np.mean(quality_ecpm))
    test_total_ecpm.append(np.mean(total_ecpm))
    test_com_bid.append(np.mean(com_bid))


fea_day = pd.DataFrame()
fea_day['ad_id'] = test_aid
fea_day['compete_bid'] = test_com_bid
fea_day['compete_pctr'] = test_com_pctr
fea_day['compete_quality_ecpm'] = test_quality_ecpm
fea_day['compete_total_ecpm'] = test_total_ecpm
fea_day.to_csv('../usingData/feature/compete_test.csv', index=False)