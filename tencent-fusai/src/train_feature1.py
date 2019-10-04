# -*- coding: utf-8 -*-
"""
@file:train_feature1.py
@time:2019/6/1 16:58
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
请求的uid的个数
统计日志文件中的，竞争list里面只需要aid就可以了，展开以后，对aid进行group by操作以后的shape就是参与竞争的数量，
对uid进行unique就是人群曝光范围
'''
total_fea = pd.DataFrame()
for na in name:
    ttt = time.time()
    print(time.localtime(ttt))
    nas = na.split('_')
    day = nas[-1]
    print(day, '  processing')
    data = pd.read_table('../metaData/metaTrain/' + na + '.out', header=None)
    compete = data[4].values
    uid_list = data[2].values

    # uid是每一条log的uid，是重复的，然后id是唯一标志一条log记录的,展开以后按照uid groupby
    # uid的唯一值取出来，其对应的shape就是log条数
    new_uid = []
    new_aid = []
    def deal(x):
        xx = x.split(',')
        t = xx[0]
        return t
    for i, ad_list in enumerate(compete):
        uid = uid_list[i]
        ads = ad_list.split(';')
        temp = list(map(deal, ads))
        new_aid.extend(temp)
        temp_uid = []
        temp_uid = [uid] * len(ads)
        new_uid.extend(temp_uid)

    ttt = time.time()
    print('log process done ', time.localtime(ttt))
    new_log = pd.DataFrame()
    new_log['ad_id'] = new_aid
    new_log['uid'] = new_uid
    group = new_log.groupby('ad_id')
    fea_aid = []
    uid_len = []
    request_len = []
    # 展开所有的日志以后，得到的新的dataframe，然后对其进行操作，就是今天的每个广告的这个特征
    for g in group:
        g1 = g[1]
        ad_id = g1['ad_id'].values[0]
        request_nums = g1.shape[0]
        uid_nums = g1['uid'].unique().shape[0]

        fea_aid.append(ad_id)
        uid_len.append(uid_nums)
        request_len.append(request_nums)
    ttt = time.time()
    print('saving begin ', time.localtime(ttt))
    fea_day = pd.DataFrame()
    fea_day['ad_id'] = fea_aid
    fea_day['request_nums'] = request_len
    fea_day['uid_nums'] = uid_len
    fea_day.loc[:, 'day'] = day
    fea_day.to_csv('../usingData/feature/' + str(day) + '_uid.csv', index=False)






