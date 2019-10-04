# -*- coding: utf-8 -*-
"""
@file:feature3_4.py
@time:2019/6/7 10:09
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import time
# request集合的大小，uid的集合数量,这些特征，应该是用全部的日志文件，不是只有曝光成功的日志文件
name = ['track_log_20190410', 'track_log_20190411', 'track_log_20190412', 'track_log_20190413', 'track_log_20190414',
        'track_log_20190415', 'track_log_20190416', 'track_log_20190417', 'track_log_20190418', 'track_log_20190419',
        'track_log_20190420', 'track_log_20190421',
        'track_log_20190422']
'''
统计日志文件中的，和每个广告竞争的top2个广告的rate,用的是当天的rate特征，但是这样肯定是一个leak特征
1-将日志展开，取出来该条日志中的两组特征，分别对应于胜出日志和其它非胜出日志的值；
2-取出来对应的aid，然后将第一组特征给第一个aid，其它的aid对应于第二组特征
3-新的dataframe进行aid的groupby操作，取均值和前75%数据的均值，存储在list中
4-以aid，ecpm和bid四个特征，day，组成新的dataframe，存储就是train的这些竞争特征
'''
using_rate = pd.read_csv('../usingData/feature/total_ad_sucess_rate.csv')
total_fea = pd.DataFrame()
continue_num = 0
for na in name:
    ttt = time.time()
    print(time.localtime(ttt))
    nas = na.split('_')
    day = nas[-1]
    print(day, '  processing')
    data = pd.read_table('../metaData/metaTrain/' + na + '.out', header=None)
    compete = data[4].values
    uid_list = data[2].values

    every_day_rate = using_rate[using_rate['day'] == int(day)]
    everyday_aid = every_day_rate['ad_id'].values
    everyday_rate = every_day_rate['sucess_rate'].values
    every_day = {}
    for i, k in enumerate(everyday_aid):
        if k not in every_day:
            every_day[k] = everyday_rate[i]
    every_day[-1] = 0.0

    # 取出对应的aid
    def deal(x):
        xx = x.split(',')
        t = xx[0]
        return t

    com_rate = []
    new_aid = []
    def fx(x):
        t = x
        xx = t.split(',')
        if int(xx[-2]) == 1:
            t = -100
        if float(xx[3]) < 0:
            t = -100
        return t
    for i, ad_list in enumerate(compete):

        # 先把被过滤的广告找出来
        adss = ad_list.split(';')
        ads_temp = list(map(fx, adss))
        temp_data = pd.DataFrame()
        temp_data['t'] = ads_temp
        temp_data = temp_data[temp_data['t'] != -100]
        ads = temp_data['t'].values

        # 取出来aid
        if len(ads) < 3:
            continue_num += 1
            continue
        temp = list(map(deal, ads))
        new_aid.extend(temp)

        # 取出两组特征值备用
        aid1 = ads[0].split(',')
        aid2 = ads[1].split(',')
        aid3 = ads[2].split(',')

        ad_id1 = int(aid1[0])
        ad_id2 = int(aid2[0])
        ad_id3 = int(aid3[0])

        if ad_id1 not in every_day:
            ad_id1 = -1
        if ad_id2 not in every_day:
            ad_id2 = -1
        if ad_id3 not in every_day:
            ad_id3 = -1
        rate1 = every_day[ad_id1]
        rate2 = every_day[ad_id2]
        rate3 = every_day[ad_id3]

        temp_rate1 = (rate2 + rate3)/2
        temp_rate2 = (rate2 + rate1)/2

        # 分别将两组特征给对应的aid
        com_rate.append(temp_rate1)
        t_rate = [temp_rate2] * (len(ads) - 1)
        com_rate.extend(t_rate)

    print(continue_num)
    ttt = time.time()
    print('log process done ', time.localtime(ttt))
    new_log = pd.DataFrame()
    new_log['ad_id'] = new_aid
    new_log['compete_rate'] = com_rate

    group = new_log.groupby('ad_id')
    fea_rate = []
    fea_aid = []

    # 展开所有的日志以后，得到的新的dataframe，然后对其进行操作，就是今天的每个广告的这个特征
    for g in group:
        g1 = g[1]
        ad_id = g1['ad_id'].values[0]
        fea_aid.append(ad_id)
        bid_list = g1['compete_rate'].values
        fea_rate.append(np.mean(bid_list))

    ttt = time.time()
    print('saving begin ', time.localtime(ttt))
    fea_day = pd.DataFrame()
    fea_day['ad_id'] = fea_aid
    fea_day['compete_rate'] = fea_rate
    fea_day.loc[:, 'day'] = day
    fea_day.to_csv('../usingData/feature/' + str(day) + '_compete_rate.csv', index=False)