# -*- coding: utf-8 -*-
"""
@file:make_new_ad_cross_rate.py
@time:2019/6/11 10:29
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
test = pd.read_csv('../usingData/test/test_bid.csv')
print(test.shape)
rate = pd.read_csv('../usingData/feature/everyday_exposure_train.csv')
cross_train = pd.read_csv('../usingData/feature/cross_fea_train.csv')
cross_test = pd.read_csv('../usingData/feature/cross_fea_test.csv')

# 制作新建广告和旧的广告
operate = pd.read_csv('../usingData/train/train_bid.csv')
print(operate)


def f(x):
    xx = str(x)
    tt = xx[0:8]
    t = int(tt)
    return t


changeTime = operate['changeTime'].values
new_time = list(map(f, changeTime))
operate['changeDay'] = new_time
operate.rename(columns={'changeDay': 'day'}, inplace=True)

train0 = pd.merge(cross_train, operate, on=['ad_id', 'day'], how='left')
train0 = train0[train0['operateType'] != 2]
train = train0

rate1 = rate[['ad_id', 'sucess_rate', 'day']]
print(train0.shape)
train_use = pd.merge(train0, rate1, on=['ad_id', 'day'], how='left')
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

for fea in count_fea:
    print(fea + " label feature preparing")
    print(len(cross_test[fea].unique()))
    train_res, test_res, id_res, reqday_res = one_feature_exposure3(train_use, cross_test[fea].values, fea, date)
    for i, cols in enumerate(col_name1):
        col = fea + '_' + cols
        train_use2[col] = train_res[:, i]
        test_use2[col] = test_res[:, i]

train_use2['ad_id'] = id_res[:train_use.shape[0]]
train_use2['day'] = reqday_res[:train_use.shape[0]]
print(train_use2.shape)
print(test_use2.shape)
test_use2['ad_id'] = test['ad_id']
train_use2.to_csv('../usingData/feature/old_ad_cross_rate_train.csv', index=False)
test_use2.to_csv('../usingData/feature/old_ad_cross_rate_test4.csv', index=False)