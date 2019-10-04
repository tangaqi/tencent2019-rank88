# -*- coding: utf-8 -*-
"""
@file:make_cross_rate_feature.py
@time:2019/6/9 8:39
@author:Tangj
@software:Pycharm
@Desc
"""

import pandas as pd
from utils import *

date = ['20190410', '20190411', '20190412', '20190413', '20190414',
        '20190415', '20190416', '20190417', '20190418', '20190419',
        '20190420', '20190421', '20190422']
# rate = pd.read_csv('../usingData/feature/rate_expose.csv')
rate = pd.read_csv('../usingData/feature/everyday_exposure_train.csv')
cross_train = pd.read_csv('../usingData/sub_totaltrain/cross_fea_train.csv')
cross_test = pd.read_csv('../usingData/feature/cross_fea_test.csv')

rate1 = rate[['ad_id', 'sucess_rate', 'day']]
print(cross_train.shape)
train_use = pd.merge(cross_train, rate1, on=['ad_id', 'day'], how='left')
train_use = train_use.fillna(0)
print(train_use.shape)
col_name1 = ['max_rate', 'min_rate', 'mean_rate', 'median_rate']
train_use2 = pd.DataFrame()
test_use2 = pd.DataFrame()
count_fea = ['ad_industry_id|ad_count_id', 'ad_industry_id|goods_type', 'ad_industry_id|ad_size_mean',
             'ad_count_id|goods_type', 'ad_count_id|ad_size_mean', 'goods_id|goods_type', 'goods_id|ad_size_mean',
             'ad_industry_id|ad_count_id|goods_type', 'ad_industry_id|ad_count_id|ad_size_mean',
             'ad_industry_id|goods_type|ad_size_mean', 'ad_industry_id|ad_count_idcount',
             'ad_industry_id|goods_typecount', 'ad_industry_id|ad_size_meancount', 'ad_count_id|goods_typecount',
             'ad_count_id|ad_size_meancount', 'goods_id|goods_typecount', 'goods_id|ad_size_meancount',
             'ad_industry_id|ad_count_id|goods_typecount', 'ad_industry_id|ad_count_id|ad_size_meancount',
             'ad_industry_id|goods_type|ad_size_meancount']

# count_fea = ['ad_id']

for fea in count_fea:
    print(fea + " label feature preparing")
    print(len(cross_test[fea].unique()))
    train_res, test_res, id_res, reqday_res = one_feature_exposure3(train_use, cross_test[fea].values, fea, date)
    for i, cols in enumerate(col_name1):
        col = fea + '_' + cols
        # train_use2[col] = train_res[:, i]
        test_use2[col] = test_res[:, i]

# train_use2['ad_id'] = id_res[:train_use.shape[0]]
# train_use2['day'] = reqday_res[:train_use.shape[0]]
# print(train_use2.shape)
# print(test_use2.shape)
# train_use2.to_csv('../usingData/feature/aid_rate_train.csv', index=False)
print(test_use2)
test_use2.to_csv('../usingData/feature/aid_rate_test.csv', index=False)