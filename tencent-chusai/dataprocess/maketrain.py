# -*- coding: utf-8 -*-
"""
@file:maketrain.py
@time:2019/5/6 16:42
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import gc
import time

name = ['log_0_1999', 'log_2000_3999', 'log_4000_5999','log_6000_7999', 'log_8000_9999', 'log_10000_19999',
        'log_20000_29999', 'log_30000_39999','log_40000_49999',
        'log_50000_59999','log_60000_69999','log_70000_79999','log_80000_89999','log_90000_99999',
        'log_100000_109999','log_110000_119999','log_120000_129999','log_130000_139999']


def group_split(list_values):
    new_values = []
    for values in list_values:
        vals = values.split(',')
        for i in vals:
            if i not in new_values:
                new_values.append(i)
    new_values.sort()
    if 'all' in new_values:
        return 'all'

    str_val = new_values[0]
    flag = 1
    for i in new_values:
        if flag == 1:
            str_val = str(i)
            flag = 0
        else:
            str_val = str_val + ',' + str(i)
    return str_val


def putting_time_process(put_time):
    bi_val = [0] * 48
    for time in put_time:
        time = int(time)
        bi_time = bin(time)
        j = 0
        num = len(bi_time) - 1
        while num > 1:
            bi_val[j] += int(bi_time[num])
            num -= 1
            j += 1
    n = 47
    flag = 1
    times = '0'
    total = 0
    while n >= 0:
        if bi_val[n] >= 1:
            val = 1
            total += 1
        else:
            val = 0
        if flag == 1:
            flag = 0
            times = str(val)
        else:
            times = times + str(val)
        n -= 1
    re_times1 = int(times, 2)

    return re_times1, times, total

def disstatus(train, option):
    print("status processing")
    distime = []
    opstatus = option[option['changeField'] == 1]
    opstatus.index = opstatus['statime']
    opstatus.sort_index()
    opstatus.index = range(opstatus.shape[0])
    values = opstatus['changeValue']
    optime = opstatus['statime'].values
    flag = 1
    j = 0
    for i in values:
        if (i == '0') & (flag == 1):
            distime.append(optime[j])
            flag = 0
        if (i == '1') & (flag == 0):
            distime.append(optime[j])
            flag = 1
        j += 1
    j = 0
    if len(distime) == 0:
        return train
    elif (len(distime) % 2 == 0):
        for i in range(int(len(distime) / 2)):
            Train = pd.DataFrame()
            t1 = distime[j]
            t2 = distime[j + 1]
            # print(t1)
            # print(t2)
            j += 2
            train1 = train[train['statime'] < t1]
            # print(train1['Reqday'].unique())
            Train = pd.concat([Train, train1])
            train1 = train[train['statime'] > t2]
            # print(train1['Reqday'].unique())
            Train = pd.concat([Train, train1])
            train = Train
    else:
        t1 = distime[-1]
        train = train[train['statime'] < t1]
        Train = pd.DataFrame()
        for i in range(int(len(distime) / 2)):
            Train = pd.DataFrame()
            t1 = distime[j]
            t2 = distime[j + 1]
            j += 2
            train1 = train[train['statime'] < t1]
            Train = pd.concat([Train, train1])
            train2 = train[train['statime'] > t2]
            Train = pd.concat([Train, train2])
            train = Train
        #     print(train.shape)
    del Train
    gc.collect()
    return train


def initValue(train, operate):
    print("initing processing")
    ope = operate[operate['optionType'] == 2]
    # 初始化bid
    print("initing bid")
    inb = ope[ope['changeField'] == 2]['changeValue']
    if inb.shape[0] == 0:
        train.loc[:, 'adBid'] = 88
    else:
        inbid = '-1'
        for i in inb:
            inbid = i
            break
        train.loc[:, 'adBid'] = int(inbid)
    # 初始化人群
    print("initing peo")
    train.loc[:, 'age'] = 'all'
    train.loc[:, 'gender'] = 'all'
    train.loc[:, 'area'] = 'all'
    train.loc[:, 'status'] = 'all'
    train.loc[:, 'education'] = 'all'
    train.loc[:, 'consuptionAbility'] = 'all'
    train.loc[:, 'device'] = 'all'
    train.loc[:, 'work'] = 'all'
    train.loc[:, 'connectionType'] = 'all'
    train.loc[:, 'behavior'] = 'all'
    if ope[ope['changeField'] == 3].shape[0] != 0:
        inpeo = ope[ope['changeField'] == 3]['changeValue'].values[0]
        peofea = inpeo.split("|")
        for fea in peofea:
            l = fea.split(':')
            if (len(l) < 2):
                continue
            feas = l[1].split(',')
            feas.sort()
            if (feas is None):
                continue
            flags = 1
            feature = '0'
            for i in feas:
                if (flags == 1):
                    feature = str(i)
                    flags = 0
                    continue
                feature = feature + ',' + str(i)
                #         feature = str(feas)
            if l[0].lower() == 'age':
                if (len(feas) < 100):
                    # print(feature)
                    train.loc[:, 'age'] = feature
            if l[0].lower() == 'gender':
                # print(feature)
                train.loc[:, 'gender'] = feature
            if l[0].lower() == 'area':
                # print(feature)
                train.loc[:, 'area'] = feature
            if l[0].lower() == 'status':
                # print(feature)
                train.loc[:, 'status'] = feature
            if l[0].lower() == 'education':
                # print(feature)
                train.loc[:, 'education'] = feature
            if l[0].lower() == 'consuptionability':
                # print(feature)
                train.loc[:, 'consuptionAbility'] = feature
            if l[0].lower() == 'os':
                # print(feature)
                train.loc[:, 'device'] = feature
            if l[0].lower() == 'work':
                # print(feature)
                train.loc[:, 'work'] = feature
            if l[0].lower() == 'connectiontype':
                # print(feature)
                train.loc[:, 'connectionType'] = feature
            if l[0].lower() == 'behavior':
                # print(feature)
                train.loc[:, 'behavior'] = feature
    # 初始化投放时间
    inti = ope[ope['changeField'] == 4]['changeValue'].values[0]
    putting = inti.split(',')
    len_inti = ope[ope['changeField'] == 4].shape[0]
    #     print(putting)
    if (len(putting) != 7) or (len_inti == 0):
        train.loc['puttingTime'] = '281474976710655'
    else:
        train.loc[train['week'] == 0, 'puttingTime'] = putting[0]
        train.loc[train['week'] == 1, 'puttingTime'] = putting[1]
        train.loc[train['week'] == 2, 'puttingTime'] = putting[2]
        train.loc[train['week'] == 3, 'puttingTime'] = putting[3]
        train.loc[train['week'] == 4, 'puttingTime'] = putting[4]
        train.loc[train['week'] == 5, 'puttingTime'] = putting[5]
        train.loc[train['week'] == 6, 'puttingTime'] = putting[6]

    return train


def changeBid(train, operate):
    print("changebid processing")
    option = operate[operate['optionType'] == 1]
    opbid = option[option['changeField'] == 2]
    if opbid.shape[0] == 0:
        return train

    opbid.index = opbid['statime']
    opbid.sort_index()
    opbid.index = range(opbid.shape[0])
    values = opbid['changeValue']
    optime = opbid['statime'].values
    j = 0
    for ti in optime:
        Train = pd.DataFrame()
        train1 = train[train['statime'] <= ti]
        # print(ti)
        # print(train1['Reqday'].unique())
        Train = pd.concat([Train, train1])
        train2 = train[train['statime'] > ti]
        # print(train2['Reqday'].unique())
        train2.loc[:, 'adBid'] = int(values[j])
        #         print(train2.shape)
        Train = pd.concat([Train, train2])
        train = Train
        j += 1
    del Train
    gc.collect()
    print(train.shape)
    return train


def changePeo(train, operate):
    option = operate[operate['optionType'] == 1]
    opbid = option[option['changeField'] == 3]
    if opbid.shape[0] == 0:
        return train
    print("changepeo processing")
    opbid.index = opbid['statime']
    opbid.sort_index()
    opbid.index = range(opbid.shape[0])
    values = opbid['changeValue']
    optime = opbid['statime'].values
    j = 0
    x = 1
    for ti in optime:
        Train = pd.DataFrame()
        train1 = train[train['statime'] <= ti]
        # print(ti)
        # print(train1['Reqday'].unique())
        Train = pd.concat([Train, train1])
        train2 = train[train['statime'] > ti]
        # print(train2['Reqday'].unique())
        # 人群重定向之前，需要重新给出初始化操作
        train2.loc[:, 'age'] = 'all'
        train2.loc[:, 'gender'] = 'all'
        train2.loc[:, 'area'] = 'all'
        train2.loc[:, 'status'] = 'all'
        train2.loc[:, 'education'] = 'all'
        train2.loc[:, 'consuptionAbility'] = 'all'
        train2.loc[:, 'device'] = 'all'
        train2.loc[:, 'work'] = 'all'
        train2.loc[:, 'connectionType'] = 'all'
        train2.loc[:, 'behavior'] = 'all'
        #         print(values[j])
        inp = values[j]
        for i in inp:
            inpeo = str(i)
            break
        peofea = inpeo.split("|")
        for fea in peofea:
            l = fea.split(':')
            if (len(l) < 2):
                continue
            feas = l[1].split(',')
            feas.sort()
            if (feas is None):
                continue
            #             feature = str(feas)
            flags = 1
            feature = '0'
            for i in feas:
                if (flags == 1):
                    feature = str(i)
                    flags = 0
                    continue
                feature = feature + ',' + str(i)
            if l[0].lower() == 'age':
                if (len(feas) < 100):
                    train2.loc[:, 'age'] = feature
            if l[0].lower() == 'gender':
                train2.loc[:, 'gender'] = feature
            if l[0].lower() == 'area':
                train2.loc[:, 'area'] = feature
            if l[0].lower() == 'status':
                train2.loc[:, 'status'] = feature
            if l[0].lower() == 'education':
                train2.loc[:, 'education'] = feature
            if l[0].lower() == 'consuptionability':
                train2.loc[:, 'consuptionAbility'] = feature
            if l[0].lower() == 'os':
                train2.loc[:, 'device'] = feature
            if l[0].lower() == 'work':
                train2.loc[:, 'work'] = feature
            if l[0].lower() == 'Connectiontype':
                train2.loc[:, 'connectionType'] = feature
            if l[0].lower() == 'behavior':
                train2.loc[:, 'behavior'] = feature
        Train = pd.concat([Train, train2])
        train = Train
        j += 1
    del Train
    gc.collect()
    print(train.shape)
    return train


def changeTime(train, operate):
    print("changeTime processing")
    option = operate[operate['optionType'] == 1]
    opbid = option[option['changeField'] == 4]
    if opbid.shape[0] == 0:
        return train
    opbid.index = opbid['statime']
    opbid.sort_index()
    opbid.index = range(opbid.shape[0])
    values = opbid['changeValue'].values
    optime = opbid['statime'].values
    if len(values) == 0:
        return train
    j = 0
    for ti in optime:
        Train = pd.DataFrame()
        train1 = train[train['statime'] <= ti]
        Train = pd.concat([Train, train1])
        train2 = train[train['statime'] > ti]
        putting = values[j].split(',')
        if (len(putting) == 7):
            train2.loc[train2['week'] == 0, 'puttingTime'] = putting[0]
            train2.loc[train2['week'] == 1, 'puttingTime'] = putting[1]
            train2.loc[train2['week'] == 2, 'puttingTime'] = putting[2]
            train2.loc[train2['week'] == 3, 'puttingTime'] = putting[3]
            train2.loc[train2['week'] == 4, 'puttingTime'] = putting[4]
            train2.loc[train2['week'] == 5, 'puttingTime'] = putting[5]
            train2.loc[train2['week'] == 6, 'puttingTime'] = putting[6]
        Train = pd.concat([Train, train2])
        train = Train
        j += 1
    del Train
    gc.collect()
    print(train.shape)
    return train


option = pd.read_csv('../data/adoption_use3.csv')
x = []
for i in option['statime'].values:
    m = i.split('_')
    if len(m) == 4:
        mon = m[0].zfill(2)
        day = m[1].zfill(2)
        h = m[2].zfill(2)
        mii = m[3].zfill(2)
        x.append(mon + day + h + mii)
    else:
        x.append('0')

option['statime'] = x

adstatic = pd.read_csv('../data/ad_static_feature.csv')
option.fillna('-1')
adstatic.fillna('-1')
mask = ~((option['changeTime'] == 0) & (option['optionType'] != 2))
adoption = option[mask]
TotalTrain = pd.DataFrame()
for na in name:
    column_name = ['Reqday', 'ad_id', 'ad_count_id', 'goods_id', 'goods_type', 'create_time',
                   'ad_industry_id', 'ad_size', 'adBid', 'age', 'gender',
                   'education', 'consuptionAbility', 'device', 'connectionType', 'work', 'area',
                   'status', 'behavior', 'puttingTime ', 'Reqday', 'exposure', 'user_id']
    totalTrain = pd.DataFrame()
    logdata = pd.read_csv('../data/logdel/' + str(na) + '.csv')
    userdata = pd.read_csv('../data/user/user_' + str(na) + '.csv')
    createdata = pd.read_csv('../data/havecreate.csv')
    creAid = createdata['have'].unique()

    # 先将日期和小时信息打进去
    # 将每次的option时间以“|”拼起来存储，作为不同广告的区分，也作为之后明确生效时间段划分数据
    day = []
    hour = []
    optiontime = []
    day_hour = []
    weekday = []
    for i in logdata['adRequestTime']:
        tt = time.localtime(i)
        mons = str(tt.tm_mon).zfill(2)
        days = str(tt.tm_mday).zfill(2)
        hou = str(tt.tm_hour).zfill(2)
        mmi = str(tt.tm_min).zfill(2)
        day.append(str(mons + '_' + days))
        hour.append(str(hou + '_' + mmi))
        d_h = str(mons + days + hou + mmi)
        day_hour.append(str(d_h))
        weekday.append(tt.tm_wday)
    logdata['Reqday'] = day
    logdata['week'] = weekday
    logdata['Reqhourmin'] = hour
    logdata['statime'] = day_hour

    # 对每一个ad的train进行遍历，构造训练集
    ad = logdata['ad_id'].unique()
    for aid in ad:
        if aid not in creAid:
            continue
        aidlog = logdata[logdata['ad_id'] == aid]
        aidlog = pd.merge(aidlog, adstatic, on='ad_id', how='left')
        aidlog = pd.merge(aidlog, userdata, on='user_id', how='left')
        aidlog = aidlog.fillna('-1')
        train = aidlog.copy()
        option1 = adoption[adoption['ad_id'] == aid]
        ad_create_time = aidlog['create_time'].unique()[0]
        tt = time.localtime(ad_create_time)
        mons = str(tt.tm_mon).zfill(2)
        days = str(tt.tm_mday).zfill(2)
        hou = str(tt.tm_hour).zfill(2)
        mmi = str(tt.tm_min).zfill(2)
        cre_time = str(mons + days + hou + mmi)
        # print(cre_time)
        op1 = option1[option1['statime'] >= cre_time]
        op2 = option1[option1['optionType'] == 2]
        option = pd.concat([op1, op2])
        train = disstatus(train, option)

        if (train.shape[0] == 0):
            continue
        train = initValue(train, option)
        train = changeBid(train, option)
        train = changePeo(train, option)
        train = changeTime(train, option)
        grouped = train.groupby(['Reqday', 'ad_id', 'adBid', 'puttingTime'])
        for i in grouped:
            log = i[1][0:1]
            log_all = i[1]
            train_i = pd.DataFrame()
            train_i['Reqday'] = [i[0][0]]
            train_i['ad_id'] = [i[0][1]]
            train_i['exposure'] = i[1].shape[0]
            # 对于广告的动态特征，这里就先对其进行一个group，放在同一个list里面
            train_i['adBid'] = np.mean(log_all['adBid'].values)  # 取均值
            time_int, time_two, time_total = putting_time_process(log_all['puttingTime'].unique())
            train_i['puttingTime_int'] = time_int
            train_i['puttingTime_two'] = time_two  # 这是转换为二进制的48位的01串，直接作为48个编码的特征使用
            train_i['puttingTime_total'] = time_total  # 这是总的投放时长，直接作为一个数值特征使用
            train_i['age'] = group_split(log_all['age'].values)  # 直接取一个list，原始的取出来是一个字符串，先在重新对其取并再生成新的字符串
            train_i['gender'] = group_split(log_all['gender'].values)
            train_i['area'] = group_split(log_all['area'].values)
            train_i['status'] = group_split(log_all['status'].values)
            train_i['education'] = group_split(log_all['education'].values)
            train_i['consuptionAbility'] = group_split(log_all['consuptionAbility'].values)
            train_i['device'] = group_split(log_all['device'].values)
            train_i['work'] = group_split(log_all['work'].values)
            train_i['connectionType'] = group_split(log_all['connectionType'].values)
            train_i['behavior'] = group_split(log_all['behavior'].values)
            # 广告静态特征，可以直接赋值，认为是在日志里面是不变化的
            train_i['ad_count_id'] = log['ad_count_id'].values[0]
            train_i['goods_id'] = log['goods_id'].values[0]
            train_i['goods_type'] = log['goods_type'].values[0]
            train_i['create_time'] = log['create_time'].values[0]
            train_i['ad_industry_id'] = log['ad_industry_id'].values[0]
            train_i['ad_size'] = log['adSize'].values[0]
            train_i['user_id'] = str(i[1]['user_id'].unique())
            train_i['adPosition_id'] = str(i[1]['adPosition_id'].unique())

            train_i['adPctr'] = np.mean(log['adPctr'].values)
            train_i['adQuality_ecpm'] = np.mean(log['adQuality_ecpm'].values)
            train_i['totalEcpm'] = np.mean(log['totalEcpm'].values)

            totalTrain = pd.concat([totalTrain, train_i])
    TotalTrain = pd.concat([TotalTrain, totalTrain])
    del totalTrain
    gc.collect()
TotalTrain.to_csv('TotalTrain_cpc.csv', index=False)