# -*- coding: utf-8 -*-
"""
@file:train_feature2.py
@time:2019/6/5 13:54
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
竞争队列中的pctr、ecpm、bid等信息的均值，用的是日志文件中竞争队列里面当天的pctr等信息
统计日志文件中的，和每个广告竞争的top2个广告的total_ecpm, quality_ecpm, pctr, bid四个值
1-将日志展开，取出来该条日志中的两组特征，分别对应于胜出日志和其它非胜出日志的值；
2-取出来对应的aid，然后将第一组特征给第一个aid，其它的aid对应于第二组特征
3-新的dataframe进行aid的groupby操作，取均值和前75%数据的均值，存储在list中
4-以aid，ecpm和bid四个特征，day，组成新的dataframe，存储就是train的这些竞争特征
'''
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

    # 取出对应的aid
    def deal(x):
        xx = x.split(',')
        t = xx[0]
        return t

    com_pctr = []
    quality_ecpm = []
    total_ecpm = []
    com_bid = []
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
        temp_bid1 = (float(aid2[1]) + float(aid3[1]))/2
        temp_pctr1 = (float(aid2[2]) + float(aid3[2]))/2
        temp_qu_ecpm1 = (float(aid2[3]) + float(aid3[3]))/2
        temp_to_ecpm1 = (float(aid2[4]) + float(aid3[4]))/2
        temp_bid2 = (float(aid2[1]) + float(aid1[1])) / 2
        temp_pctr2 = (float(aid2[2]) + float(aid1[2])) / 2
        temp_qu_ecpm2 = (float(aid2[3]) + float(aid1[3])) / 2
        temp_to_ecpm2 = (float(aid2[4]) + float(aid1[4])) / 2
        # print(temp_to_ecpm2)
        # print(temp_to_ecpm1)

        # 分别将两组特征给对应的aid
        com_bid.append(temp_bid1)
        com_pctr.append(temp_pctr1)
        quality_ecpm.append(temp_qu_ecpm1)
        total_ecpm.append(temp_to_ecpm1)

        t_bid = [temp_bid2] * (len(ads) - 1)
        t_pctr = [temp_pctr2] * (len(ads) - 1)
        t_qu_ecpm = [temp_qu_ecpm2] * (len(ads) - 1)
        t_to_ecpm = [temp_to_ecpm2] * (len(ads) - 1)
        com_bid.extend(t_bid)
        com_pctr.extend(t_pctr)
        quality_ecpm.extend(t_qu_ecpm)
        total_ecpm.extend(t_to_ecpm)

    print(continue_num)
    ttt = time.time()
    print('log process done ', time.localtime(ttt))
    new_log = pd.DataFrame()
    new_log['ad_id'] = new_aid
    new_log['bid'] = com_bid
    new_log['pctr'] = com_pctr
    new_log['quality_ecpm'] = quality_ecpm
    new_log['total_ecpm'] = total_ecpm
    group = new_log.groupby('ad_id')
    fea_bid = []
    fea_pctr = []
    fea_qu_ecpm = []
    fea_to_ecpm = []
    fea_aid = []

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
    fea_day['compete_bid'] = fea_bid
    fea_day['compete_pctr'] = fea_pctr
    fea_day['compete_quality_ecpm'] = fea_qu_ecpm
    fea_day['compete_total_ecpm'] = fea_to_ecpm
    fea_day.loc[:, 'day'] = day
    fea_day.to_csv('../usingData/feature/com/' + str(day) + '_compete.csv', index=False)