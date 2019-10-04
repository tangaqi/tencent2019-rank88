# -*- coding: utf-8 -*-
"""
@file:test_feature3.py
@time:2019/6/7 8:56
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
using_rate = pd.read_csv('../usingData/feature/total_ad_sucess_rate.csv')
log = log.fillna(-100)

'''
统计竞争队列里面的rate特征，用的是22号的rate
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

every_day_rate = using_rate[using_rate['day'] == 20190422]
everyday_aid = every_day_rate['ad_id'].values
everyday_rate = every_day_rate['sucess_rate'].values

every_day = {}
for i, k in enumerate(everyday_aid):
    if k not in every_day:
        every_day[k] = everyday_rate[i]
every_day[-1] = 0.0
test_com_rate = []
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
    com_rate = []
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
        ad_id1 = int(aid1[0])
        ad_id2 = int(aid2[0])
        if int(ad_id1) not in every_day:
            ad_id1 = -1
        if int(ad_id2) not in every_day:
            ad_id2 = -1
        rate1 = every_day[ad_id1]
        rate2 = every_day[ad_id2]
        rate_ave = (rate1 + rate2)/2
        com_rate.append(rate_ave)
    # 直接将取出来的数组，取均值
    test_com_rate.append(np.mean(com_rate))

fea_day = pd.DataFrame()
fea_day['ad_id'] = test_aid
fea_day['compete_rate'] = test_com_rate
fea_day.to_csv('../usingData/feature/compete_rate_test.csv', index=False)