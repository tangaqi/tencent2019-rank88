# -*- coding: utf-8 -*-
"""
@file:maketrain.py
@time:2019/5/26 14:19
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import matplotlib.pyplot as plt # 画图的库
import warnings
warnings.filterwarnings('ignore') # 忽略警告信息
import numpy as np
import seaborn as sns
import time
sns.set(color_codes=True)
np.random.seed(sum(map(ord, 'distributions')))
flags = 0
name = ['track_log_20190410', 'track_log_20190411', 'track_log_20190412', 'track_log_20190413', 'track_log_20190414',
        'track_log_20190415', 'track_log_20190416', 'track_log_20190417', 'track_log_20190418', 'track_log_20190419',
        'track_log_20190420', 'track_log_20190421',
        'track_log_20190422']
log = pd.DataFrame()
static_fea = pd.read_table('../metaData/map_ad_static.out', header=None)
cols = ['ad_id', 'createTime', 'ad_count_id', 'goods_id', 'goods_type', 'ad_industry_id', 'ad_size']
static = pd.DataFrame()
for i, k in enumerate(cols):
    static[k] = static_fea[i]
def group_split(list_values):

    flag = 1
    for i in list_values:
        if flag == 1:
            str_val = str(i)
            flag = 0
        else:
            str_val = str_val + ',' + str(i)
    return str_val


for na in name:
    nas = na.split('_')
    day = nas[-1]
    print(day)
    data0 = pd.read_csv('../usingData/log/' + na + '_0.csv')
    data1 = pd.read_csv('../usingData/log/' + na + '_1.csv')
    data2 = pd.read_csv('../usingData/log/' + na + '_2.csv')
    data3 = pd.read_csv('../usingData/log/' + na + '_3.csv')
    data4 = pd.read_csv('../usingData/log/' + na + '_4.csv')

    data = pd.concat([data0, data1])
    data = pd.concat([data, data2])
    data = pd.concat([data, data3])
    data = pd.concat([data, data4])
    # ['RequestId', 'RequestTime', 'uid', 'PositionId', 'ad_id', 'bid', 'pctr',
    #        'quality_ecpm', 'total_ecpm', 'competeAd']
    ex = data.groupby('ad_id')
    aid_new = []
    pctr_new = []
    quality_ecpm = []
    total_ecpm = []
    bid_new = []
    for exi in ex:
        log_i = exi[1]
        log_i = log_i[log_i['quality_ecpm'] >= 0.0]
        if log_i.shape[0] == 0:
            continue
        aid = log_i['ad_id'].values[0]
        aid_new.append(aid)
        pctr = log_i['uid'].values
        uid = group_split(pctr)
        pctr_new.append(uid)

    pctr_fea = pd.DataFrame()
    pctr_fea['uid'] = pctr_new
    pctr_fea['ad_id'] = aid_new
    pctr_fea.loc[:, 'day'] = day
    print(log.shape)
    log = pd.concat([log, pctr_fea])
    if flags == 0:
        print(log)
        flags = 1

log.to_csv('../usingData/train/uid_train.csv', index=False)








