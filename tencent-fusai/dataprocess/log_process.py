# -*- coding: utf-8 -*-
"""
@file:log_process.py
@time:2019/5/25 21:57
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np

name = ['track_log_20190410', 'track_log_20190411', 'track_log_20190412', 'track_log_20190413', 'track_log_20190414',
        'track_log_20190415', 'track_log_20190416', 'track_log_20190417', 'track_log_20190418', 'track_log_20190419',
        'track_log_20190420', 'track_log_20190421',
        'track_log_20190422']
operate = pd.read_table('../metaData/metaTrain/final_map_bid_opt.out', header=None)
cpcId = operate[0].unique()
for na in name:
    log = pd.read_table('../metaData/metaTrain/' + na + '.out', header=None)
    col = ['RequestId', 'RequestTime', 'uid', 'PositionId', 'competeAd']
    logAll = pd.DataFrame()
    for i, k in enumerate(col):
        logAll[k] = log[i]
    compete = logAll['competeAd'].values
    new = []
    for i, k in enumerate(compete):
        tem = logAll[i:i+1]
        ad = k.split(';')
        for j in ad:
            fea = j.split(',')
            if fea[-1] == '1':
                if int(fea[0]) not in cpcId:
                    continue
                temp = []
                temp = [tem['RequestId'].values[0], tem['RequestTime'].values[0], tem['uid'].values[0],
                        tem['PositionId'].values[0], tem['competeAd'].values[0]]
                temp.extend(fea[:6])
                new.append(temp)
    l = int(len(new)/5)
    print(len(new))
    for m in range(5):
        if m == 4:
            end = len(new) + 1
        else:
            end = (1 + m) * l
        news = new[m * l: end]
        news = np.array(news)
        print(news.shape)
        cols = ['RequestId', 'RequestTime', 'uid', 'PositionId', 'competeAd', 'ad_id', 'bid', 'pctr',
            'quality_ecpm', 'total_ecpm']
        logNew = pd.DataFrame()
        for i, k in enumerate(cols):
            if i == 4:
                continue
            logNew[k] = news[:, i]
        logNew['competeAd'] = news[:, 4]
        flag = 1
        if flag == 1:
#             print(logNew)
            flag = 0
        logNew.to_csv('../usingData/log/' + na + '_' + str(m) + '.csv', index=False)
