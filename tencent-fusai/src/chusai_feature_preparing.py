# -*- coding: utf-8 -*-
"""
@file:chusai_feature_preparing.py
@time:2019/6/1 22:01
@author:Tangj
@software:Pycharm
@Desc
"""
import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from time import time
import math
from sklearn.metrics import *
import pandas as pd
from utils import *
# part = int(sys.argv[1])
# 预定义各类list
one_fea_statistic = ['ad_count_id', 'goods_id', 'goods_type',
               'ad_industry_id','ad_id', 'ad_size']
date = ['20190410', '20190411', '20190412', '20190413', '20190414',
        '20190415', '20190416', '20190417', '20190418', '20190419',
        '20190420', '20190421', '20190422']
print("reading train data")
train = pd.read_csv('../usingData/train/sub_total_train.csv')
train = train.fillna(-1)
print(train.shape)
print("reading test data")
test = pd.read_csv('../usingData/test/test_bid.csv')
static = pd.read_csv('../usingData/train/ad_static.csv')
print(test.columns)
print(static.columns)
test = pd.merge(test, static, on='ad_id', how='left')

train['ad_size_mean'], train['ad_size_max'], train['ad_size_min'] = del_adSize(train['ad_size'].values)
test['ad_size_mean'], test['ad_size_max'], test['ad_size_min'] = del_adSize(test['ad_size'].values)
print(train.info())
print(train)

def part1():
    '''
    返回的是统计特征，也就是各个特征出现的频次。对于多值特征，返回的是平均出现次数和出现的最大的次数
    '''

    print('statistics_fea preparing')
    train_use = pd.DataFrame()
    test_use = pd.DataFrame()
    test_use['bid'] = test['bid'].values

    for fea in one_fea_statistic:
        print(fea + ' statistic count preparing')
        col = fea + 'count'
        train_res, test_res = count_one_feature_times(train, test[fea].values, fea)
        print(train_use.shape)
        print(train_res.shape)
        train_use[col] = train_res
        test_use[col] = test_res

    train_use['ad_id'] = train['ad_id'].values
    train_use['day'] = train['day'].values
    test_use['ad_id'] = test['ad_id']
    test_use['bid'] = test['bid']
    train_use.to_csv('../usingData/sub_totaltrain/count_fea_train.csv', index = False)
    test_use.to_csv('../usingData/sub_totaltrain/count_fea_test.csv', index = False)

def part2():
    # 对转化率特征进行计算，复赛没有再用上这个特征
    train_use = pd.DataFrame()
    val_use = pd.DataFrame()
    test_use = pd.DataFrame()
    id_res = None
    reqday_res = None
    col_name1 = ['max_exposure', 'min_exposure', 'mean_exposure', 'median_exposure']
    col_name2 = ['max_bid', 'min_bid', 'mean_bid', 'median_bid']
    col_name3 = ['max_pctr', 'min_pctr', 'mean_pctr', 'median_pctr']
    col_name4 = ['max_ecpm', 'min_ecpm', 'mean_ecpm', 'median_ecpm']
    # time_name.extend(one_fea_statistic)
    # for fea in time_name:
    for fea in ['ad_count_id', 'ad_id', 'goods_id', 'ad_industry_id', 'goods_type', 'ad_size_mean']:

        print(fea + " label feature preparing")
        train_res, val_res, test_res,train_res2, val_res2, test_res2, id_res, reqday_res = one_feature_exposure2(train, val,
                                                                                            test[fea].values, fea, date)
        train_res_pctr, val_res_pctr, test_res_pctr = one_feature_pctr2(train, val,
                                                                           test[fea].values, fea, date, 'adPctr')
        train_res_ecpm, val_res_ecpm, test_res_ecpm = one_feature_pctr2(train, val,
                                                                       test[fea].values, fea, date, 'adQuality_ecpm')
        for i, cols in enumerate(col_name1):
            col = fea + '_' + cols
            train_use[col] = train_res[:, i]
            val_use[col] = val_res[:, i]
            test_use[col] = test_res[:, i]
        for i, cols in enumerate(col_name2):
            col = fea + '_' + cols
            train_use[col] = train_res2[:, i]
            val_use[col] = val_res2[:, i]
            test_use[col] = test_res2[:, i]
        for i, cols in enumerate(col_name3):
            col = fea + '_' + cols
            train_use[col] = train_res_pctr[:, i]
            val_use[col] = val_res_pctr[:, i]
            test_use[col] = test_res_pctr[:, i]
        for i, cols in enumerate(col_name4):
            col = fea + '_' + cols
            train_use[col] = train_res_ecpm[:, i]
            val_use[col] = val_res_ecpm[:, i]
            test_use[col] = test_res_ecpm[:, i]

    train_use['ad_id'] = id_res[:train.shape[0]]
    train_use['Reqday'] = reqday_res[:train.shape[0]]
    val_use['ad_id'] = id_res[-val.shape[0]:]
    val_use['Reqday'] = reqday_res[-val.shape[0]:]
    test_use['ad_id'] = test['ad_id']
    test_use['adBid'] = test['adBid']
    train_use.to_csv('../data/train/one-hot/testB/X_train_2.csv', index=False)
    val_use.to_csv('../data/train/one-hot/testB/X_val_2.csv', index=False)
    test_use.to_csv('../data/train/one-hot/testB/X_test_2.csv', index=False)


def part3():
    # 交叉特征的生成，和交叉特征对应的count特征
    train_use = pd.DataFrame()
    test_use = pd.DataFrame()
    id_res = None
    reqday_res = None
    train['ad_size_mean'], train['ad_size_max'], train['ad_size_min'] = del_adSize(train['ad_size'].values)
    test['ad_size_mean'], test['ad_size_max'], test['ad_size_min'] = del_adSize(test['ad_size'].values)
    f1 = 'ad_industry_id'
    count_fea = []
    for f2 in ['ad_count_id', 'goods_type', 'ad_size_mean']:
        col = f1 + '|' + f2
        count_fea.append(col)
        train_use[col], test_use[col] = one_hot_feature_concat(train, test, f1, f2)
    f1 = 'ad_count_id'
    for f2 in ['goods_type', 'ad_size_mean']:
        col = f1 + '|' + f2
        count_fea.append(col)
        train_use[col], test_use[col] = one_hot_feature_concat(train, test, f1, f2)
    f1 = 'goods_id'
    for f2 in ['goods_type', 'ad_size_mean']:
        col = f1 + '|' + f2
        count_fea.append(col)
        train_use[col], test_use[col] = one_hot_feature_concat(train, test, f1, f2)

    train_use['goods_type'] = train['goods_type']
    test_use['goods_type'] = test['goods_type']
    train_use['ad_size_mean'] = train['ad_size_mean']
    test_use['ad_size_mean'] = test['ad_size_mean']
    f1 = 'ad_industry_id|ad_count_id'
    for f2 in ['goods_type', 'ad_size_mean']:
        col = f1 + '|' + f2
        count_fea.append(col)
        train_use[col], test_use[col] = one_hot_feature_concat(train_use, test_use, f1, f2)
    f1 = 'ad_industry_id|goods_type'
    for f2 in ['ad_size_mean']:
        col = f1 + '|' + f2
        count_fea.append(col)
        train_use[col], test_use[col] = one_hot_feature_concat(train_use, test_use, f1, f2)
    train_use.drop(['ad_size_mean', 'goods_type'], axis=1, inplace=True)
    test_use.drop(['ad_size_mean', 'goods_type'], axis=1, inplace=True)

    # 对交叉特征求统计特征
    for fea in count_fea:
        print(fea + ' statistic preparing')
        col = fea + 'count'
        train_res, test_res = count_one_feature_times(train_use, test_use[fea].values, fea)
        train_use[col] = train_res
        test_use[col] = test_res

    train_use['ad_id'] = train['ad_id'].values
    train_use['day'] = train['day'].values

    test_use['ad_id'] = test['ad_id'].values
    test_use['bid'] = test['bid'].values
    train_use.to_csv('../usingData/feature/cross_fea_train.csv', index=False)
    test_use.to_csv('../usingData/feature/cross_fea_test.csv', index=False)

def part4():
    # 对转化率特征进行计算，复赛的转化率特征不再是用这个函数生成
    train_use = pd.DataFrame()
    val_use = pd.DataFrame()
    test_use = pd.DataFrame()
    id_res = None
    reqday_res = None
    col_name1 = ['max_exposure', 'min_exposure', 'mean_exposure', 'median_exposure']
    col_name2 = ['max_bid', 'min_bid', 'mean_bid', 'median_bid']
    col_name3 = ['max_pctr', 'min_pctr', 'mean_pctr', 'median_pctr']
    col_name4 = ['max_ecpm', 'min_ecpm', 'mean_ecpm', 'median_ecpm']
    # time_name.extend(one_fea_statistic)
    # for fea in time_name:
    for fea in ['area','age']:
        print(fea + "label feature preparing")
        train_res, val_res, test_res, train_res2, val_res2, test_res2 = \
            vector_feature_exposure(train, val, test[fea].values, fea, date)
        train_pctr, val_pctr, test_pctr = vector_feature_pctr(train, val, test[fea].values, fea, date,'adPctr')
        train_ecpm, val_ecpm, test_ecpm = vector_feature_pctr(train, val, test[fea].values, fea, date,'adQuality_ecpm')
        for i, cols in enumerate(col_name1):
            col = fea + '_' + cols
            train_use[col] = train_res[:, i]
            val_use[col] = val_res[:, i]
            print(val_use.shape)
            print(val_res.shape)
            test_use[col] = test_res[:, i]
        for i, cols in enumerate(col_name2):
            col = fea + '_' + cols
            train_use[col] = train_res2[:, i]
            val_use[col] = val_res2[:, i]
            test_use[col] = test_res2[:, i]
        for i, cols in enumerate(col_name3):
            col = fea + '_' + cols
            train_use[col] = train_pctr[:, i]
            val_use[col] = val_pctr[:, i]
            test_use[col] = test_pctr[:, i]
        for i, cols in enumerate(col_name4):
            col = fea + '_' + cols
            train_use[col] = train_ecpm[:, i]
            val_use[col] = val_ecpm[:, i]
            test_use[col] = test_ecpm[:, i]

    train_use['ad_id'] = id_res[:train.shape[0]]
    train_use['Reqday'] = reqday_res[:train.shape[0]]
    val_use['ad_id'] = id_res[-val.shape[0]:]
    val_use['Reqday'] = reqday_res[-val.shape[0]:]
    test_use['ad_id'] = test['ad_id']
    test_use['adBid'] = test['adBid']
    train_use.to_csv('../data/train/one-hot/testB/X_train_muti2.csv', index=False)
    val_use.to_csv('../data/train/one-hot/testB/X_val_muti2.csv', index=False)
    test_use.to_csv('../data/train/one-hot/testB/X_test_muti2.csv', index=False)

# part1（）统计出现次数的特征，需要过滤，生成的太多了，感觉比较好的是先用方差将那些差距比较小的直接过滤掉，
# 一共有22个统计特征，好像也不是很多。。

part1()
# part2()
part3()
# part4()



