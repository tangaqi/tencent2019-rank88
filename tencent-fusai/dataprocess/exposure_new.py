# -*- coding: utf-8 -*-
"""
@file:exposure_new.py
@time:2019/6/4 9:14
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
log = pd.DataFrame()
static_fea = pd.read_table('../metaData/map_ad_static.out', header=None)
cols = ['ad_id', 'createTime', 'ad_count_id', 'goods_id', 'goods_type', 'ad_industry_id', 'ad_size']
static = pd.DataFrame()
for i, k in enumerate(cols):
    static[k] = static_fea[i]
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
    print(data.shape)
    # print(data.columns)
    mask1 = data['bid'] < 20000
    data = data[mask1]
    mask2 = data['pctr'] < 300
    data = data[mask2]
    mask3 = data['quality_ecpm'] > 0
    data = data[mask3]
    mask4 = data['quality_ecpm'] < 4000
    data = data[mask4]
    mask5 = data['total_ecpm'] < 20000
    data = data[mask5]
    print(data.shape)

    ex = data.groupby('ad_id').size().reset_index(name='ex')
    ex.loc[:, 'day'] = day
    log = pd.concat([log, ex])

log.to_csv('../usingData/train/exposure_new.csv', index=False)
# log = pd.merge(log, static, on='ad_id', how='left')
# log.to_csv('../usingData/train/metafea_train.csv', index=False)



