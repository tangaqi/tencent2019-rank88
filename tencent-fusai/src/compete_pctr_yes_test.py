# -*- coding: utf-8 -*-
"""
@file:compete_pctr_yes_test.py
@time:2019/6/12 21:10
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
统计的仍然是pctr，ecpm和bid等的均值信息，用的是前一天的平移过来的pctr等信息，这里的平移的前一天的是早就已经统计好的信息，是日志中的总的均值等信息
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
using_rate = pd.read_csv('../usingData/feature/total_ad_pctr.csv')
every_day_rate = using_rate[using_rate['day'] == 20190423]
everyday_aid = every_day_rate['ad_id'].values
everyday_bid = every_day_rate['bid'].values
everyday_pctr = every_day_rate['pctr'].values
everyday_total = every_day_rate['total_ecpm'].values
everyday_quality = every_day_rate['quality_ecpm'].values
every_bid = {}
every_pctr = {}
every_total = {}
every_quality = {}
for i, k in enumerate(everyday_aid):
    if k not in every_bid:
        every_bid[k] = everyday_bid[i]
        every_pctr[k] = everyday_pctr[i]
        every_total[k] = everyday_total[i]
        every_quality[k] = everyday_quality[i]
every_bid[-1] = 0.0
every_pctr[-1] = 0.0
every_total[-1] = 0.0
every_quality[-1] = 0.0

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
        aid1 = int(ads[index1].split(',')[0])
        aid2 = int(ads[index2].split(',')[0])
        if aid1 not in every_bid:
            aid1 = -1
        if aid2 not in every_bid:
            aid2 = -1

        temp_bid1 = (every_bid[aid2] + every_bid[aid1]) / 2
        temp_pctr1 = (every_pctr[aid2] + every_pctr[aid1]) / 2
        temp_qu_ecpm1 = (every_quality[aid2] + every_quality[aid1]) / 2
        temp_to_ecpm1 = (every_total[aid2] + every_total[aid1]) / 2
        print(temp_bid1)
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
fea_day['yestoday_compete_bid'] = test_com_bid
fea_day['yestoday_compete_pctr'] = test_com_pctr
fea_day['yestoday_compete_quality_ecpm'] = test_quality_ecpm
fea_day['yestoday_compete_total_ecpm'] = test_total_ecpm
fea_day.to_csv('../usingData/feature/compete_test.csv', index=False)