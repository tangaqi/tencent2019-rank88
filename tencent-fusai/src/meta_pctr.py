# -*- coding: utf-8 -*-
"""
@file:del_log_pctr.py
@time:2019/6/12 10:01
@author:Tangj
@software:Pycharm
@Desc
"""
'''
得到的是原始的日志文件中每天的pctr和ecpm和bid等信息
把所有的日志里面的广告的pctr和ecpm等信息都遍历一边，所有的日志包括23号的日志，
对competeAd的list进行处理，首先将其取出来，然后一个日志做一个dataframe，然后按照aid进行groupby就好了
最后存储起来对应的aid和day和对应的pctr和bid和ecpm等的均值
'''
import pandas as pd
import numpy as np
import time
# request集合的大小，uid的集合数量,这些特征，应该是用全部的日志文件，不是只有曝光成功的日志文件
name = ['track_log_20190410', 'track_log_20190411', 'track_log_20190412', 'track_log_20190413', 'track_log_20190414',
        'track_log_20190415', 'track_log_20190416', 'track_log_20190417', 'track_log_20190418', 'track_log_20190419',
        'track_log_20190420', 'track_log_20190421',
        'track_log_20190422']
total_fea = pd.DataFrame()
for na in name:
    ttt = time.time()
    print(time.localtime(ttt))
    nas = na.split('_')
    day = nas[-1]
    print(day, '  processing')
    data = pd.read_table('../metaData/metaTrain/' + na + '.out', header=None)
    compete = data[4].values
    new_aid = []
    new_bid = []
    new_pctr = []
    new_quality_ecpm = []
    new_total_ecpm = []
    def deal1(x):
        xx = x.split(',')
        t1 = int(xx[0])
        return t1
    def deal2(x):
        xx = x.split(',')
        t1 = float(xx[1])
        return t1
    def deal3(x):
        xx = x.split(',')
        t1 = float(xx[2])
        return t1
    def deal4(x):
        xx = x.split(',')
        t1 = float(xx[3])
        return t1
    def deal5(x):
        xx = x.split(',')
        t1 = float(xx[4])
        return t1
    for i, ad_list in enumerate(compete):
        ads = ad_list.split(';')
        t_aid = list(map(deal1, ads))
        t_bid = list(map(deal2, ads))
        t_pctr = list(map(deal3, ads))
        t_quality_ecpm = list(map(deal4, ads))
        t_total_ecpm = list(map(deal5, ads))
        new_aid.extend(t_aid)
        new_bid.extend(t_bid)
        new_pctr.extend(t_pctr)
        new_quality_ecpm.extend(t_quality_ecpm)
        new_total_ecpm.extend(t_total_ecpm)

    ttt = time.time()
    print('log process done ', time.localtime(ttt))
    new_log = pd.DataFrame()
    new_log['ad_id'] = new_aid
    new_log['bid'] = new_bid
    new_log['pctr'] = new_pctr
    new_log['total_ecpm'] = new_total_ecpm
    new_log['quality_ecpm'] = new_quality_ecpm
    group = new_log.groupby('ad_id')

    fea_aid = []
    fea_bid = []
    fea_pctr = []
    fea_qu_ecpm = []
    fea_to_ecpm = []

    # 展开所有的日志以后，得到的新的dataframe，然后对其进行操作，就是今天的每个广告的这个特征
    for g in group:
        g1 = g[1]
        ad_id = g1['ad_id'].values[0]
        fea_aid.append(ad_id)
        pctr_list = g1['pctr'].values
        total_ecpm_list = g1['total_ecpm'].values
        quality_ecpm_list = g1['quality_ecpm'].values
        bid_list = g1['bid'].values
        fea_bid.append(np.mean(bid_list))
        fea_pctr.append(np.mean(pctr_list))
        fea_qu_ecpm.append(np.mean(quality_ecpm_list))
        fea_to_ecpm.append(np.mean(total_ecpm_list))

    ttt = time.time()
    print('saving begin ', time.localtime(ttt))
    fea_day = pd.DataFrame()
    fea_day['ad_id'] = fea_aid
    fea_day['meta_bid'] = fea_bid
    fea_day['meta_pctr'] = fea_pctr
    fea_day['meta_quality_ecpm'] = fea_qu_ecpm
    fea_day['meta_total_ecpm'] = fea_to_ecpm
    fea_day.loc[:, 'day'] = day
    fea_day.to_csv('../usingData/feature/com/' + str(day) + '_metaecpm.csv', index=False)