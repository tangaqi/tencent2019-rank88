# -*- coding: utf-8 -*-
"""
@file:utils.py
@time:2019/6/1 21:57
@author:Tangj
@software:Pycharm
@Desc
"""
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import random
import pandas as pd


def frame_to_dict(train):
    train_dict = {}
    for col in train.columns:
        train_dict[col] = train[col].columns
    return trian_dict


def del_adSize(ad_Size):
    ad_size_mean = []
    ad_size_max = []
    ad_size_min = []
    for adSize in ad_Size:
        if not isinstance(adSize, str):
            # print(adSize)
            ad_size_mean.append(adSize)
            ad_size_max.append(adSize)
            ad_size_min.append(adSize)
            continue
        size = adSize.split(',')
        s = []
        for i in size:
            s.append(int(i))
        ad_size_mean.append(np.mean(s))
        ad_size_max.append(np.max(s))
        ad_size_min.append(np.min(s))
    return ad_size_mean, ad_size_max, ad_size_max


def write_data_into_parts(data, root_path, nums=5100000):
    l = data.shape[0] // nums
    for i in range(l + 1):
        begin = i * nums
        end = min(nums * (i + 1), data.shape[0])
        t_data = data[begin:end]
        t_data.tofile(root_path + '.bin')


def write_dict(path, data):
    fw = open(path, 'w')
    for key in data:
        fw.write(str(key) + ',' + str(data[key]) + '\n')
    fw.close()


def read_allfea(path):
    f = open(path, 'r')
    fea = '0'
    for i in f:
        fea = i
    fea_val = fea.split(',')
    index_dict = {}
    for i, fea in enumerate(fea_val):
        index_dict[fea] = i + 1
    if '-1' not in index_dict:
        index_dict['-1'] = len(fea_val)
    return fea, index_dict


def one_hot_feature_concat(train, test, fea1, fea2, filter_num=100):
    train1 = train[fea1].values
    train2 = train[fea2].values
    test1 = test[fea1].values
    test2 = test[fea2].values

    train_data = []
    test_data = []
    train_res = []
    test_res = []
    for i, values in enumerate(train1):
        new = str(values) + '|' + str(train2[i])
        train_data.append(new)
    for i, values in enumerate(test1):
        new = str(values) + '|' + str(test2[i])
        # print(new)
        test_data.append(new)
    count_dict = {}
    for d in train_data:
        if d not in count_dict:
            count_dict[d] = 0
        count_dict[d] += 1
    filter_set = []
    for i in count_dict:
        if count_dict[i] < 1:
            filter_set.append(i)

    index_dict = {}
    begin_index = 1
    for d in train_data:
        # 给出现的value赋予一个index指引
        if d in filter_set:
            d = '-1'
        if d not in index_dict:
            index_dict[d] = begin_index
            begin_index += 1
        train_res.append(index_dict[d])
    if '-1' not in index_dict:
        index_dict['-1'] = begin_index
    for d in test_data:
        if d not in index_dict or d in filter_set:
            d = '-1'
        test_res.append(index_dict[d])
    print(test_res)
    return np.array(train_res), np.array(test_res)


def one_hot_feature_process(train_data, val_data, test2_data, begin_num, filter_num=0):
    index_dict = {}
    begin_index = begin_num
    train_res = []
    for d in train_data:
        #         print(d)
        # 给出现的value赋予一个index指引
        if d not in index_dict:
            index_dict[d] = begin_index
            begin_index += 1
        #             print(index_dict[d])
        train_res.append(index_dict[d])
    if '-1' not in index_dict:
        index_dict['-1'] = begin_index
    val_res = []
    for d in val_data:
        if d not in index_dict:
            index_dict[d] = begin_index
            begin_index += 1
        val_res.append(index_dict[d])
    test2_res = []
    for d in test2_data:
        if d not in index_dict:
            d = '-1'
        test2_res.append(index_dict[d])
    #     print(np.array(train_res))
    return np.array(train_res), np.array(val_res), np.array(test2_res), index_dict


def vector_feature_process(train_data, val_data, test2_data, begin_num, max_len, index_dict):
    train_res = []
    train_res2 = []
    val_res2 = []
    test2_res2 = []

    train_rate = []
    val_rate = []
    test2_rate = []

    for d in train_data:
        lx = d.split(',')
        row = [0] * max_len
        row2 = [0] * max_len
        if len(lx) > max_len or d == 'all':
            j = 0
            for i in index_dict:
                if j >= max_len:
                    break
                row[j] = index_dict[i]
                j += 1
            train_res.append(row)
            row2 = [1] * max_len
            train_res2.append(row2)
            train_rate.append(1)
            continue
        for i, x in enumerate(lx):
            if x not in index_dict:
                x = '-1'
            row[i] = index_dict[x]
            row2[row[i]] = 1
        train_res.append(row)
        train_res2.append(row2)
        train_rate.append(len(lx) / max_len)

    val_res = []
    for d in val_data:
        lx = d.split(',')
        row = [0] * max_len
        row2 = [0] * max_len
        if len(lx) > max_len or d == 'all':
            j = 0
            for i in index_dict:
                if j >= max_len:
                    break
                row[j] = index_dict[i]
                j += 1
            val_res.append(row)
            row2 = [1] * max_len
            val_res2.append(row2)
            val_rate.append(1)
            continue
        for i, x in enumerate(lx):
            if x not in index_dict:
                x = '-1'
            row[i] = index_dict[x]
            row2[row[i]] = 1
        val_res.append(row)
        val_res2.append(row2)
        val_rate.append(len(lx) / max_len)

    test2_res = []
    for d in test2_data:
        lx = d.split(',')
        row = [0] * max_len
        row2 = [0] * max_len
        if len(lx) > max_len or d == 'all':
            j = 0
            for i in index_dict:
                if j >= max_len:
                    break
                row[j] = index_dict[i]
                j += 1
            test2_res.append(row)
            row2 = [1] * max_len
            test2_res2.append(row2)
            test2_rate.append(1)
            continue
        for i, x in enumerate(lx):
            if x not in index_dict:
                x = '-1'
            row[i] = index_dict[x]
            row2[row[i]] = 1
        test2_res.append(row)
        test2_res2.append(row2)
        test2_rate.append(len(lx) / max_len)
    return np.array(train_res), np.array(val_res), np.array(test2_res), index_dict, np.array(train_res2), np.array(
        val_res2), np.array(test2_res2), np.array(train_rate), np.array(val_rate), np.array(test2_rate),


def count_one_feature_times(train, test, fea):
    count_dict = {}
    test_res = []
    train_res = []
    for val in train[fea].values:
        if val not in count_dict:
            count_dict[val] = 0
        count_dict[val] += 1
    if '-1' not in count_dict:
        count_dict['-1'] = 1
    for i in train[fea].values:
        train_res.append(count_dict[i])
    for i in test:
        if i not in count_dict:
            i = '-1'
        test_res.append(count_dict[i])

    return np.array(train_res), np.array(test_res)


def count_vector_feature_times(train, val_data, test, fea):
    count_dict = {}
    val_res = []
    test_res = []
    train_res = []
    Train = pd.concat([train, val_data])
    for val in Train[fea].values:
        vals = val.split(',')
        for i in vals:
            if i not in count_dict:
                count_dict[i] = 0
            count_dict[i] += 1
    if '-1' not in count_dict:
        count_dict['-1'] = 1

    for val in train[fea].values:
        vals = val.split(',')
        l = []
        for i in vals:
            l.append(count_dict[i])
        # ['max', 'mean', 'min', 'median']
        max_l = np.max(l)
        mean_l = np.mean(l)
        min_l = np.min(l)
        median_l = np.median(l)
        train_res.append([max_l, mean_l, min_l, median_l])
    for val in val_data[fea].values:
        vals = val.split(',')
        l = []
        for i in vals:
            l.append(count_dict[i])
        # ['max', 'mean', 'min', 'median']
        max_l = np.max(l)
        mean_l = np.mean(l)
        min_l = np.min(l)
        median_l = np.median(l)
        val_res.append([max_l, mean_l, min_l, median_l])
    for val in test:
        vals = val.split(',')
        l = []
        for i in vals:
            if i not in count_dict:
                i = '-1'
            l.append(count_dict[i])
        # ['max', 'mean', 'min', 'median']
        max_l = np.max(l)
        mean_l = np.mean(l)
        min_l = np.min(l)
        median_l = np.median(l)
        test_res.append([max_l, mean_l, min_l, median_l])

    return np.array(train_res), np.array(val_res), np.array(test_res)


# 对曝光、pctr和ecpm和bid的特征

def one_feature_exposure2(Train, test, fea, date):
    # 返回曝光的最大值，最小值，均值，中位数四个值，
    # 返回bid的最大值，最小值，均值，中位数四个值，
    test_res = []
    train_res = []
    id_res = []
    reqday_res = []
    train = Train
    num1 = train[train['day'] == 20190410].shape[0]
    id_res.extend(train[train['day'] == 20190410]['ad_id'].values)
    reqday_res.extend(train[train['day'] == 20190410]['day'].values)
    for i in range(num1):
        train_res.append([0, 0, 0, 0])
    for i in range(len(date) - 1):
        day = int(date[i + 1])
        train_compute = Train[Train['day'] == day]
        train_count = Train[Train['day'] < day]
        id_res.extend(train_compute['ad_id'].values)
        reqday_res.extend(train_compute['day'].values)
        exposure_dict = {}
        for value in train_count[fea].values:
            if value not in exposure_dict:
                exposure_dict[value] = []
                train1 = train_count[train_count[fea] == value]['sucess_rate'].values
                exposure_dict[value].append(np.max(train1))
                exposure_dict[value].append(np.min(train1))
                exposure_dict[value].append(np.mean(train1))
                exposure_dict[value].append(np.median(train1))
        if '-1' not in exposure_dict:
            exposure_dict['-1'] = [0, 0, 0, 0]
        for value in train_compute[fea].values:
            if value not in exposure_dict:
                value = '-1'
            train_res.append(exposure_dict[value])

    train_count = Train[Train['day'] > 20190414]
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['sucess_rate'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))

    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0, 0, 0, 0]
    for value in test:
        if value not in exposure_dict:
            value = '-1'
        test_res.append(exposure_dict[value])
    return np.array(train_res), np.array(test_res), np.array(id_res), np.array(reqday_res)

def one_feature_exposure4(Train, test, fea, date):
    test_res = []
    train_res = []
    id_res = []
    reqday_res = []
    train = Train
    train_count = train[train['day'] == 20190410]
    train_compute = train[train['day'] == 20190410]
    id_res.extend(train_compute['ad_id'].values)
    reqday_res.extend(train_compute['day'].values)
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['ex'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))

    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0.9, 0.9]

    for value in train_compute[fea].values:
        if value not in exposure_dict:
            value = '-1'
        train_res.append(exposure_dict[value])

    train_count = train[train['day'] == 20190410]
    train_compute = train[train['day'] == 20190411]
    id_res.extend(train_compute['ad_id'].values)
    reqday_res.extend(train_compute['day'].values)
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['ex'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0.9, 0.9]

    for value in train_compute[fea].values:
        if value not in exposure_dict:
            value = '-1'
        train_res.append(exposure_dict[value])

    for i in range(len(date) - 2):
        day1 = int(date[i + 2])
        day2 = int(date[i + 1])
        day3 = int(date[i])

        train1 = Train[Train['day'] == day3]
        train2 = Train[Train['day'] == day2]
        train_compute = Train[Train['day'] == day1]
        id_res.extend(train_compute['ad_id'].values)
        reqday_res.extend(train_compute['day'].values)
        train_count = pd.concat([train1, train2])
        exposure_dict = {}
        for value in train_count[fea].values:
            if value not in exposure_dict:
                exposure_dict[value] = []
                train1 = train_count[train_count[fea] == value]['ex'].values
                exposure_dict[value].append(np.mean(train1))
                exposure_dict[value].append(np.median(train1))

        if '-1' not in exposure_dict:
            exposure_dict['-1'] = [0.9, 0.9]
        for value in train_compute[fea].values:
            if value not in exposure_dict:
                value = '-1'
            train_res.append(exposure_dict[value])

    train1 = train[train['day'] == 20190421]
    train2 = train[train['day'] == 20190422]
    train_count = pd.concat([train1, train2])
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            # print(train_count[train_count[fea] == value].shape[0])
            train1 = train_count[train_count[fea] == value]['ex'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0.9, 0.9]
    num_dis = 0
    for value in test:
        # print(value)
        if value not in exposure_dict:
            num_dis += 1
            value = '-1'
        test_res.append(exposure_dict[value])
    print(num_dis)
    return np.array(train_res), np.array(test_res), \
           np.array(id_res), np.array(reqday_res)

def one_feature_exposure3(Train, test, fea, date):
    # 返回曝光的最大值，最小值，均值，中位数四个值，
    # 返回bid的最大值，最小值，均值，中位数四个值，
    test_res = []
    train_res = []
    id_res = []
    reqday_res = []
    train = Train

    # train_count = train[train['day'] == 20190410]
    # train_compute = train[train['day'] == 20190410]
    # id_res.extend(train_compute['ad_id'].values)
    # reqday_res.extend(train_compute['day'].values)
    # exposure_dict = {}
    # for value in train_count[fea].values:
    #     if value not in exposure_dict:
    #         train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #         exposure_dict[value] = []
    #         exposure_dict[value].append(np.max(train1))
    #         exposure_dict[value].append(np.min(train1))
    #         exposure_dict[value].append(np.mean(train1))
    #         exposure_dict[value].append(np.median(train1))
    #
    # if '-1' not in exposure_dict:
    #     exposure_dict['-1'] = [0, 0, 0, 0]
    #
    # for value in train_compute[fea].values:
    #     if value not in exposure_dict:
    #         value = '-1'
    #     train_res.append(exposure_dict[value])
    #
    # train_count = train[train['day'] == 20190410]
    # train_compute = train[train['day'] == 20190411]
    # id_res.extend(train_compute['ad_id'].values)
    # reqday_res.extend(train_compute['day'].values)
    # exposure_dict = {}
    # for value in train_count[fea].values:
    #     if value not in exposure_dict:
    #         train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #         exposure_dict[value] = []
    #         exposure_dict[value].append(np.max(train1))
    #         exposure_dict[value].append(np.min(train1))
    #         exposure_dict[value].append(np.mean(train1))
    #         exposure_dict[value].append(np.median(train1))
    # if '-1' not in exposure_dict:
    #     exposure_dict['-1'] = [0, 0, 0, 0]
    #
    # for value in train_compute[fea].values:
    #     if value not in exposure_dict:
    #         value = '-1'
    #     train_res.append(exposure_dict[value])
    #
    # for i in range(len(date) - 2):
    #     day1 = int(date[i + 2])
    #     day2 = int(date[i + 1])
    #     day3 = int(date[i])
    #
    #     train1 = Train[Train['day'] == day3]
    #     train2 = Train[Train['day'] == day2]
    #     train_compute = Train[Train['day'] == day1]
    #     id_res.extend(train_compute['ad_id'].values)
    #     reqday_res.extend(train_compute['day'].values)
    #     train_count = pd.concat([train1, train2])
    #     exposure_dict = {}
    #     for value in train_count[fea].values:
    #         if value not in exposure_dict:
    #             exposure_dict[value] = []
    #             train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #             exposure_dict[value].append(np.max(train1))
    #             exposure_dict[value].append(np.min(train1))
    #             exposure_dict[value].append(np.mean(train1))
    #             exposure_dict[value].append(np.median(train1))
    #
    #     if '-1' not in exposure_dict:
    #         exposure_dict['-1'] = [0, 0, 0, 0]
    #     for value in train_compute[fea].values:
    #         if value not in exposure_dict:
    #             value = '-1'
    #         train_res.append(exposure_dict[value])

    # train1 = train[train['day'] == 20190421]
    train_count = train[train['day'] == 20190422]
    # train_count = pd.concat([train1, train2])
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['sucess_rate'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0, 0, 0, 0]
    num_dis = 0
    for value in test:
        # print(value)
        if value not in exposure_dict:
            num_dis += 1
            value = '-1'
        test_res.append(exposure_dict[value])
    print(num_dis)
    return np.array(train_res), np.array(test_res), \
           np.array(id_res), np.array(reqday_res)

def one_feature_exposure(train, val, test, fea, date):
    # 返回曝光的最大值，最小值，均值，中位数四个值，
    # 返回bid的最大值，最小值，均值，中位数四个值，
    val_res = []
    test_res = []
    train_res = []
    val_res2 = []
    test_res2 = []
    train_res2 = []
    train_res3 = []
    id_res = []
    reqday_res = []
    Train = pd.concat([train, val])
    num1 = train[train['Reqday'] == '02_16'].shape[0]
    id_res.extend(train[train['Reqday'] == '02_16']['ad_id'].values)
    reqday_res.extend(train[train['Reqday'] == '02_16']['Reqday'].values)
    for i in range(num1):
        train_res.append([8, 8, 8, 8])
        train_res2.append([8, 8, 8, 8])

    train_count = train[train['Reqday'] == '02_16']
    train_compute = train[train['Reqday'] == '02_17']
    id_res.extend(train_compute['ad_id'].values)
    reqday_res.extend(train_compute['Reqday'].values)
    exposure_dict = {}
    bid_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['exposure'].values
            exposure_dict[value] = []
            bid_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
            train2 = train_count[train_count[fea] == value]['adBid'].values
            bid_dict[value].append(np.max(train2))
            bid_dict[value].append(np.min(train2))
            bid_dict[value].append(np.mean(train2))
            bid_dict[value].append(np.median(train2))

    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [8, 8, 8, 8]
        bid_dict['-1'] = [8, 8, 8, 8]

    for value in train_compute[fea].values:
        if value not in exposure_dict:
            value = '-1'
        train_res.append(exposure_dict[value])
        train_res2.append(bid_dict[value])

    for i in range(len(date) - 2):
        day1 = date[i + 2]
        day2 = date[i + 1]
        day3 = date[i]

        train1 = Train[Train['Reqday'] == day3]
        train2 = Train[Train['Reqday'] == day2]
        train_compute = Train[Train['Reqday'] == day1]
        id_res.extend(train_compute['ad_id'].values)
        reqday_res.extend(train_compute['Reqday'].values)
        train_count = pd.concat([train1, train2])
        exposure_dict = {}
        bid_dict = {}
        for value in train_count[fea].values:
            if value not in exposure_dict:
                exposure_dict[value] = []
                bid_dict[value] = []
                train1 = train_count[train_count[fea] == value]['exposure'].values
                exposure_dict[value].append(np.max(train1))
                exposure_dict[value].append(np.min(train1))
                exposure_dict[value].append(np.mean(train1))
                exposure_dict[value].append(np.median(train1))
                train2 = train_count[train_count[fea] == value]['adBid'].values
                bid_dict[value].append(np.max(train2))
                bid_dict[value].append(np.min(train2))
                bid_dict[value].append(np.mean(train2))
                bid_dict[value].append(np.median(train2))
        if '-1' not in exposure_dict:
            exposure_dict['-1'] = [8, 8, 8, 8]
            bid_dict['-1'] = [8, 8, 8, 8]
        for value in train_compute[fea].values:
            if value not in exposure_dict:
                value = '-1'
            train_res.append(exposure_dict[value])
            train_res2.append(bid_dict[value])

        train_res1 = train_res[:(Train.shape[0] - val.shape[0])]
        val_res = train_res[-val.shape[0]:]
        train_res3 = train_res2[:(Train.shape[0] - val.shape[0])]
        val_res2 = train_res2[-val.shape[0]:]
    train1 = train[train['Reqday'] == '03_19']
    train2 = train[train['Reqday'] == '03_18']
    train_count = pd.concat([train1, train2])
    exposure_dict = {}
    bid_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['exposure'].values
            exposure_dict[value] = []
            bid_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
            train2 = train_count[train_count[fea] == value]['adBid'].values
            bid_dict[value].append(np.max(train2))
            bid_dict[value].append(np.min(train2))
            bid_dict[value].append(np.mean(train2))
            bid_dict[value].append(np.median(train2))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [8, 8, 8, 8]
        bid_dict['-1'] = [8, 8, 8, 8]
    for value in test:
        if value not in exposure_dict:
            value = '-1'
        test_res.append(exposure_dict[value])
        test_res2.append(bid_dict[value])
    return np.array(train_res1), np.array(val_res), np.array(test_res), np.array(train_res3), np.array(val_res2), \
           np.array(test_res2), np.array(id_res), np.array(reqday_res)


def one_feature_pctr2(train, val, test, fea, date, count_fea):
    # 返回pctr的最大值，最小值，均值，中位数四个值
    val_res = []
    test_res = []
    train_res = []
    train_res2 = []
    Train = pd.concat([train, val])
    num1 = train[train['Reqday'] == '02_16'].shape[0]
    for i in range(num1):
        train_res.append([8, 8, 8, 8])
        train_res2.append([8, 8, 8, 8])
    for i in range(len(date) - 1):
        day = date[i + 1]
        train_compute = Train[Train['Reqday'] == day]
        train_count = Train[Train['Reqday'] < day]
        exposure_dict = {}
        for value in train_count[fea].values:
            if value not in exposure_dict:
                exposure_dict[value] = []
                train1 = train_count[train_count[fea] == value][count_fea].values
                exposure_dict[value].append(np.max(train1))
                exposure_dict[value].append(np.min(train1))
                exposure_dict[value].append(np.mean(train1))
                exposure_dict[value].append(np.median(train1))
        if '-1' not in exposure_dict:
            exposure_dict['-1'] = [8, 8, 8, 8]
        for value in train_compute[fea].values:
            if value not in exposure_dict:
                value = '-1'
            train_res.append(exposure_dict[value])
        train_res1 = train_res[:(Train.shape[0] - val.shape[0])]
        val_res = train_res[-val.shape[0]:]
    # train1 = train[train['Reqday'] == '03_19']
    # train2 = train[train['Reqday'] == '03_18']
    train_count = Train
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value][count_fea].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [8, 8, 8, 8]
    for value in test:
        if value not in exposure_dict:
            value = '-1'
        test_res.append(exposure_dict[value])
    return np.array(train_res1), np.array(val_res), np.array(test_res)


def one_feature_pctr(train, val, test, fea, date, count_fea):
    # 返回pctr的最大值，最小值，均值，中位数四个值
    val_res = []
    test_res = []
    train_res = []
    train_res2 = []
    Train = pd.concat([train, val])
    num1 = train[train['Reqday'] == '02_16'].shape[0]
    for i in range(num1):
        train_res.append([8, 8, 8, 8])
        train_res2.append([8, 8, 8, 8])

    train_count = train[train['Reqday'] == '02_16']
    train_compute = train[train['Reqday'] == '02_17']

    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value][count_fea].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [8, 8, 8, 8]
    for value in train_compute[fea].values:
        if value not in exposure_dict:
            value = '-1'
        train_res.append(exposure_dict[value])
    for i in range(len(date) - 2):
        day1 = date[i + 2]
        day2 = date[i + 1]
        day3 = date[i]
        train1 = Train[Train['Reqday'] == day3]
        train2 = Train[Train['Reqday'] == day2]
        train_compute = Train[Train['Reqday'] == day1]
        train_count = pd.concat([train1, train2])
        exposure_dict = {}
        for value in train_count[fea].values:
            if value not in exposure_dict:
                exposure_dict[value] = []
                train1 = train_count[train_count[fea] == value][count_fea].values
                exposure_dict[value].append(np.max(train1))
                exposure_dict[value].append(np.min(train1))
                exposure_dict[value].append(np.mean(train1))
                exposure_dict[value].append(np.median(train1))
        if '-1' not in exposure_dict:
            exposure_dict['-1'] = [8, 8, 8, 8]
        for value in train_compute[fea].values:
            if value not in exposure_dict:
                value = '-1'
            train_res.append(exposure_dict[value])
        train_res1 = train_res[:(Train.shape[0] - val.shape[0])]
        val_res = train_res[-val.shape[0]:]
    train1 = train[train['Reqday'] == '03_19']
    train2 = train[train['Reqday'] == '03_18']
    train_count = pd.concat([train1, train2])
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value][count_fea].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [8, 8, 8, 8]
    for value in test:
        if value not in exposure_dict:
            value = '-1'
        test_res.append(exposure_dict[value])
    return np.array(train_res1), np.array(val_res), np.array(test_res)


def create_mask(value, train):
    mask = []
    # 这里的train是某个多值特征的值数组
    for i in train:
        vals = i.split(',')
        flag = 0
        for j in vals:
            if j == value:
                flag = 1
                mask.append(True)
                break
        if flag == 0:
            mask.append(False)
    return mask


def vector_feature_exposure(train, val, test, fea, date):
    # 返回曝光均值，最大值，最小值,中位数四个值，
    # 返回bid的均值，最大值，最小值,中位数四个值，
    val_res = []
    test_res = []
    train_res = []
    val_res2 = []
    test_res2 = []
    train_res2 = []
    train_res3 = []
    Train = pd.concat([train, val])
    num1 = train[train['Reqday'] == '02_16'].shape[0]
    for i in range(num1):
        train_res.append([8, 8, 8, 8])
        train_res2.append([8, 8, 8, 8])

    for i in range(len(date) - 1):
        day = date[i + 1]
        train_compute = Train[Train['Reqday'] == day]
        train_count = Train[Train['Reqday'] < day]

        exposure_max = {}
        exposure_min = {}
        exposure_mean = {}
        exposure_median = {}
        bid_max = {}
        bid_min = {}
        bid_mean = {}
        bid_median = {}
        mask_dict = {}
        for vals in train_count[fea].values:
            valss = vals.split(',')
            for value in valss:
                if value not in exposure_max:
                    if value not in mask_dict:
                        mask_dict[value] = create_mask(value, train_count[fea].values)
                    train1 = train_count[mask_dict[value]]['exposure'].values
                    exposure_max[value] = np.max(train1)
                    exposure_min[value] = np.min(train1)
                    exposure_mean[value] = np.mean(train1)
                    exposure_median[value] = np.median(train1)
                    train2 = train_count[mask_dict[value]]['adBid'].values
                    bid_max[value] = np.max(train2)
                    bid_min[value] = np.min(train2)
                    bid_mean[value] = np.mean(train2)
                    bid_median[value] = np.median(train2)

        if '-1' not in exposure_max:
            exposure_max['-1'] = 8
            bid_max['-1'] = 8
        if '-1' not in exposure_min:
            exposure_min['-1'] = 8
            bid_min['-1'] = 8
        if '-1' not in exposure_mean:
            exposure_mean['-1'] = 8
            bid_mean['-1'] = 8
        if '-1' not in exposure_median:
            exposure_median['-1'] = 8
            bid_median['-1'] = 8

        for vals in train_compute[fea].values:
            max_list = []
            min_list = []
            mean_list = []
            median_list = []
            max_list2 = []
            min_list2 = []
            mean_list2 = []
            median_list2 = []
            valuess = vals.split(',')
            for value in valuess:
                if value not in exposure_max:
                    value = '-1'
                max_list.append(exposure_max[value])
                min_list.append(exposure_min[value])
                mean_list.append(exposure_mean[value])
                median_list.append(exposure_median[value])
                max_list2.append(bid_max[value])
                min_list2.append(bid_min[value])
                mean_list2.append(bid_mean[value])
                median_list2.append(bid_median[value])

            max1 = np.max(max_list)
            min1 = np.min(min_list)
            mean1 = np.mean(mean_list)
            median1 = np.median(median_list)
            train_res.append([max1, min1, mean1, median1])
            max1 = np.max(max_list2)
            min1 = np.min(min_list2)
            mean1 = np.mean(mean_list2)
            median1 = np.median(median_list2)
            train_res2.append([max1, min1, mean1, median1])

        train_res1 = train_res[:(Train.shape[0] - val.shape[0])]
        val_res = train_res[-val.shape[0]:]
        train_res3 = train_res2[:(Train.shape[0] - val.shape[0])]
        val_res2 = train_res2[-val.shape[0]:]

    train_count = Train
    exposure_max = {}
    exposure_min = {}
    exposure_mean = {}
    exposure_median = {}
    bid_max = {}
    bid_min = {}
    bid_mean = {}
    bid_median = {}
    mask_dict = {}
    for vals in train_count[fea].values:
        valss = vals.split(',')
        for value in valss:
            if value not in exposure_max:
                if value not in mask_dict:
                    mask_dict[value] = create_mask(value, train_count[fea].values)
                train1 = train_count[mask_dict[value]]['exposure'].values
                exposure_max[value] = np.max(train1)
                exposure_min[value] = np.min(train1)
                exposure_mean[value] = np.mean(train1)
                exposure_median[value] = np.median(train1)
                train2 = train_count[mask_dict[value]]['adBid'].values
                bid_max[value] = np.max(train2)
                bid_min[value] = np.min(train2)
                bid_mean[value] = np.mean(train2)
                bid_median[value] = np.median(train2)
    if '-1' not in exposure_max:
        exposure_max['-1'] = 8
        bid_max['-1'] = 8
    if '-1' not in exposure_min:
        exposure_min['-1'] = 8
        bid_min['-1'] = 8
    if '-1' not in exposure_mean:
        exposure_mean['-1'] = 8
        bid_mean['-1'] = 8
    if '-1' not in exposure_median:
        exposure_median['-1'] = 8
        bid_median['-1'] = 8
    for vals in test:
        max_list = []
        min_list = []
        mean_list = []
        median_list = []
        max_list2 = []
        min_list2 = []
        mean_list2 = []
        median_list2 = []
        valuess = vals.split(',')
        for value in valuess:
            if value not in exposure_max:
                value = '-1'
            max_list.append(exposure_max[value])
            min_list.append(exposure_min[value])
            mean_list.append(exposure_mean[value])
            median_list.append(exposure_median[value])
            max_list2.append(bid_max[value])
            min_list2.append(bid_min[value])
            mean_list2.append(bid_mean[value])
            median_list2.append(bid_median[value])
        max1 = np.max(max_list)
        min1 = np.min(min_list)
        mean1 = np.mean(mean_list)
        median1 = np.median(median_list)
        test_res.append([max1, min1, mean1, median1])
        max1 = np.max(max_list2)
        min1 = np.min(min_list2)
        mean1 = np.mean(mean_list2)
        median1 = np.median(median_list2)
        test_res2.append([max1, min1, mean1, median1])

    return np.array(train_res1), np.array(val_res), np.array(test_res), np.array(train_res3), \
           np.array(val_res2), np.array(test_res2)


def vector_feature_pctr(train, val, test, fea, date, count_fea):
    # 返回曝光均值，最大值，最小值,中位数四个值，
    val_res = []
    test_res = []
    train_res = []
    train_res2 = []
    Train = pd.concat([train, val])
    num1 = train[train['Reqday'] == '02_16'].shape[0]
    for i in range(num1):
        train_res.append([8, 8, 8, 8])
        train_res2.append([8, 8, 8, 8])

    for i in range(len(date) - 1):
        day = date[i + 1]
        train_compute = Train[Train['Reqday'] == day]
        train_count = Train[Train['Reqday'] < day]
        exposure_max = {}
        exposure_min = {}
        exposure_mean = {}
        exposure_median = {}
        bid_max = {}
        bid_min = {}
        bid_mean = {}
        bid_median = {}
        mask_dict = {}
        for vals in train_count[fea].values:
            valss = vals.split(',')
            for value in valss:
                if value not in exposure_max:
                    if value not in mask_dict:
                        mask_dict[value] = create_mask(value, train_count[fea].values)
                    train1 = train_count[mask_dict[value]][count_fea].values
                    exposure_max[value] = np.max(train1)
                    exposure_min[value] = np.min(train1)
                    exposure_mean[value] = np.mean(train1)
                    exposure_median[value] = np.median(train1)
        if '-1' not in exposure_max:
            exposure_max['-1'] = 8
            bid_max['-1'] = 8
        if '-1' not in exposure_min:
            exposure_min['-1'] = 8
            bid_min['-1'] = 8
        if '-1' not in exposure_mean:
            exposure_mean['-1'] = 8
            bid_mean['-1'] = 8
        if '-1' not in exposure_median:
            exposure_median['-1'] = 8
            bid_median['-1'] = 8

        for vals in train_compute[fea].values:
            max_list = []
            min_list = []
            mean_list = []
            median_list = []
            valuess = vals.split(',')
            for value in valuess:
                if value not in exposure_max:
                    value = '-1'
                max_list.append(exposure_max[value])
                min_list.append(exposure_min[value])
                mean_list.append(exposure_mean[value])
                median_list.append(exposure_median[value])

            max1 = np.max(max_list)
            min1 = np.min(min_list)
            mean1 = np.mean(mean_list)
            median1 = np.median(median_list)
            train_res.append([max1, min1, mean1, median1])

        train_res1 = train_res[:(Train.shape[0] - val.shape[0])]
        val_res = train_res[-val.shape[0]:]
    train_count = Train
    exposure_max = {}
    exposure_min = {}
    exposure_mean = {}
    exposure_median = {}
    mask_dict = {}
    for vals in train_count[fea].values:
        valss = vals.split(',')
        for value in valss:
            if value not in exposure_max:
                if value not in mask_dict:
                    mask_dict[value] = create_mask(value, train_count[fea].values)
                train1 = train_count[mask_dict[value]][count_fea].values
                exposure_max[value] = np.max(train1)
                exposure_min[value] = np.min(train1)
                exposure_mean[value] = np.mean(train1)
                exposure_median[value] = np.median(train1)
    if '-1' not in exposure_max:
        exposure_max['-1'] = 8
        bid_max['-1'] = 8
    if '-1' not in exposure_min:
        exposure_min['-1'] = 8
        bid_min['-1'] = 8
    if '-1' not in exposure_mean:
        exposure_mean['-1'] = 8
        bid_mean['-1'] = 8
    if '-1' not in exposure_median:
        exposure_median['-1'] = 8
        bid_median['-1'] = 8
    for vals in test:
        max_list = []
        min_list = []
        mean_list = []
        median_list = []
        valuess = vals.split(',')
        for value in valuess:
            if value not in exposure_max:
                value = '-1'
            max_list.append(exposure_max[value])
            min_list.append(exposure_min[value])
            mean_list.append(exposure_mean[value])
            median_list.append(exposure_median[value])
        max1 = np.max(max_list)
        min1 = np.min(min_list)
        mean1 = np.mean(mean_list)
        median1 = np.median(median_list)
        test_res.append([max1, min1, mean1, median1])

    return np.array(train_res1), np.array(val_res), np.array(test_res)


def one_feature_exposure5(Train, test, fea, date, new_day):
    # 返回曝光的最大值，最小值，均值，中位数四个值，
    # 返回bid的最大值，最小值，均值，中位数四个值，
    test_res = []
    train_res = []
    id_res = []
    reqday_res = []
    train = Train
    # train_count = train[train['day'] == 20190410]
    # train_compute = train[train['day'] == 20190410]
    # id_res.extend(train_compute['ad_id'].values)
    # reqday_res.extend(train_compute['day'].values)
    # exposure_dict = {}
    # for value in train_count[fea].values:
    #     if value not in exposure_dict:
    #         train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #         exposure_dict[value] = []
    #         exposure_dict[value].append(np.max(train1))
    #         exposure_dict[value].append(np.min(train1))
    #         exposure_dict[value].append(np.mean(train1))
    #         exposure_dict[value].append(np.median(train1))
    #
    # if '-1' not in exposure_dict:
    #     exposure_dict['-1'] = [0, 0, 0, 0]
    #
    # for value in train_compute[fea].values:
    #     if value not in exposure_dict:
    #         value = '-1'
    #     train_res.append(exposure_dict[value])
    #
    # train_count = train[train['day'] == 20190410]
    # train_compute = train[train['day'] == 20190411]
    # id_res.extend(train_compute['ad_id'].values)
    # reqday_res.extend(train_compute['day'].values)
    # exposure_dict = {}
    # for value in train_count[fea].values:
    #     if value not in exposure_dict:
    #         train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #         exposure_dict[value] = []
    #         exposure_dict[value].append(np.max(train1))
    #         exposure_dict[value].append(np.min(train1))
    #         exposure_dict[value].append(np.mean(train1))
    #         exposure_dict[value].append(np.median(train1))
    # if '-1' not in exposure_dict:
    #     exposure_dict['-1'] = [0, 0, 0, 0]
    #
    # for value in train_compute[fea].values:
    #     if value not in exposure_dict:
    #         value = '-1'
    #     train_res.append(exposure_dict[value])
    #
    # for i in range(len(date) - 2):
    #     day1 = int(date[i + 2])
    #     day2 = int(date[i + 1])
    #     day3 = int(date[i])
    #
    #     train1 = Train[Train['day'] == day3]
    #     train2 = Train[Train['day'] == day2]
    #     train_compute = Train[Train['day'] == day1]
    #     id_res.extend(train_compute['ad_id'].values)
    #     reqday_res.extend(train_compute['day'].values)
    #     train_count = pd.concat([train1, train2])
    #     exposure_dict = {}
    #     for value in train_count[fea].values:
    #         if value not in exposure_dict:
    #             exposure_dict[value] = []
    #             train1 = train_count[train_count[fea] == value]['sucess_rate'].values
    #             exposure_dict[value].append(np.max(train1))
    #             exposure_dict[value].append(np.min(train1))
    #             exposure_dict[value].append(np.mean(train1))
    #             exposure_dict[value].append(np.median(train1))
    #
    #     if '-1' not in exposure_dict:
    #         exposure_dict['-1'] = [0, 0, 0, 0]
    #     for value in train_compute[fea].values:
    #         if value not in exposure_dict:
    #             value = '-1'
    #         train_res.append(exposure_dict[value])

    # train_count = pd.concat([train1, train2])
    train_count = train[train['day'] >= new_day]
    exposure_dict = {}
    for value in train_count[fea].values:
        if value not in exposure_dict:
            train1 = train_count[train_count[fea] == value]['sucess_rate'].values
            exposure_dict[value] = []
            exposure_dict[value].append(np.max(train1))
            exposure_dict[value].append(np.min(train1))
            exposure_dict[value].append(np.mean(train1))
            exposure_dict[value].append(np.median(train1))
    if '-1' not in exposure_dict:
        exposure_dict['-1'] = [0, 0, 0, 0]
    num_dis = 0
    for value in test:
        if value not in exposure_dict:
            num_dis += 1
            value = '-1'
        test_res.append(exposure_dict[value])
    print(num_dis)
    return np.array(train_res), np.array(test_res), \
           np.array(id_res), np.array(reqday_res)