# -*- coding: utf-8 -*-
"""
@file:compete_pctr_yestoday.py
@time:2019/6/12 14:53
@author:Tangj
@software:Pycharm
@Desc
"""
'''
统计的仍然是pctr，ecpm和bid等的均值信息，用的是前一天的平移过来的pctr等信息，这里的平移的前一天的是早就已经统计好的信息，是日志中的总的均值等信息
'''
import pandas as pd
import numpy as np
import time
# request集合的大小，uid的集合数量,这些特征，应该是用全部的日志文件，不是只有曝光成功的日志文件
name = ['track_log_20190410', 'track_log_20190411', 'track_log_20190412', 'track_log_20190413', 'track_log_20190414',
        'track_log_20190415', 'track_log_20190416', 'track_log_20190417', 'track_log_20190418', 'track_log_20190419',
        'track_log_20190420', 'track_log_20190421',
        'track_log_20190422']
using_rate = pd.read_csv('../usingData/feature/total_ad_pctr.csv')
total_fea = pd.DataFrame()
continue_num = 0
flag = 0
for na in name:
    ttt = time.time()
    print(time.localtime(ttt))
    nas = na.split('_')
    day = nas[-1]
    print(day, '  processing')
    data = pd.read_table('../metaData/metaTrain/' + na + '.out', header=None)
    compete = data[4].values
    uid_list = data[2].values
    day2 = day
    if flag == 0:
        day = int(day)
        flag = 1
    else:
        day = int(day) - 1
    every_day_rate = using_rate[using_rate['day'] == int(day)]
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

    # 取出对应的aid
    def deal(x):
        xx = x.split(',')
        t = xx[0]
        return t

    com_bid = []
    com_pctr = []
    com_quality = []
    com_total = []
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

        if ad_id1 not in every_bid:
            ad_id1 = -1
        if ad_id2 not in every_bid:
            ad_id2 = -1
        if ad_id3 not in every_bid:
            ad_id3 = -1

        bid1 = every_bid[ad_id1]
        bid2 = every_bid[ad_id2]
        bid3 = every_bid[ad_id3]
        com_bid.append((bid2 + bid3)/2)
        temp_bid = (bid2 + bid1)/2
        t_bid = [temp_bid] * (len(ads) - 1)
        com_bid.extend(t_bid)

        bid1 = every_pctr[ad_id1]
        bid2 = every_pctr[ad_id2]
        bid3 = every_pctr[ad_id3]
        com_pctr.append((bid3 + bid2)/2)
        temp_pctr = (bid2 + bid1) / 2
        t_bid = [temp_pctr] * (len(ads) - 1)
        com_pctr.extend(t_bid)

        bid1 = every_total[ad_id1]
        bid2 = every_total[ad_id2]
        bid3 = every_total[ad_id3]
        com_total.append((bid2 + bid3)/2)
        temp_bid = (bid2 + bid1) / 2
        t_bid = [temp_bid] * (len(ads) - 1)
        com_total.extend(t_bid)

        bid1 = every_quality[ad_id1]
        bid2 = every_quality[ad_id2]
        bid3 = every_quality[ad_id3]
        com_quality.append((bid2 + bid3)/2)
        temp_bid = (bid2 + bid1) / 2
        # com_quality.append()
        t_bid = [temp_bid] * (len(ads) - 1)
        com_quality.extend(t_bid)

    print(continue_num)
    ttt = time.time()
    print('log process done ', time.localtime(ttt))
    new_log = pd.DataFrame()
    new_log['ad_id'] = new_aid
    new_log['bid'] = com_bid
    new_log['pctr'] = com_bid
    new_log['quality_ecpm'] = com_quality
    new_log['total_ecpm'] = com_total

    group = new_log.groupby('ad_id')
    fea_bid = []
    fea_pctr = []
    fea_quality = []
    fea_total = []
    fea_aid = []

    # 展开所有的日志以后，得到的新的dataframe，然后对其进行操作，就是今天的每个广告的这个特征
    for g in group:
        g1 = g[1]
        ad_id = g1['ad_id'].values[0]
        fea_aid.append(ad_id)
        bid_list = g1['bid'].values
        fea_bid.append(np.mean(bid_list))
        bid_list = g1['pctr'].values
        fea_pctr.append(np.mean(bid_list))
        bid_list = g1['quality_ecpm'].values
        fea_quality.append(np.mean(bid_list))
        bid_list = g1['total_ecpm'].values
        fea_total.append(np.mean(bid_list))


    ttt = time.time()
    print('saving begin ', time.localtime(ttt))
    fea_day = pd.DataFrame()
    fea_day['ad_id'] = fea_aid
    fea_day['yestoday_compete_bid'] = fea_bid
    fea_day['yestoday_compete_pctr'] = fea_pctr
    fea_day['yestoday_compete_quality_ecpm'] = fea_quality
    fea_day['yestoday_compete_total_ecpm'] = fea_total
    fea_day.loc[:, 'day'] = day2
    fea_day.to_csv('../usingData/feature/com/' + str(day2) + 'yes_compete_rate.csv', index=False)