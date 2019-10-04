# -*- coding: utf-8 -*-
"""
@file:make_option_type_feature.py
@time:2019/6/9 9:09
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np

'''
先给trian集中加入charge_type和'target_type'，然后把训练集按照aid进行groupby操作，
取出aid，然后将对该广告的操作数据取出来，然后按照changetime进行排序，
然后对于大于等于请求日期之后的就将其归为该条操作数据的类型
这样也不用去管对应的是新建操作还是修改操作了，反正修改肯定是在新建之后的操作。
'''

train = pd.read_csv('../usingData/train/metafea_train.csv')
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
new = operate.groupby('ad_id')
print(operate.columns)
# ['ad_id', 'changeTime', 'operateType', 'target_type', 'charge_type',
#        'bid', 'changeDay']
# for i in new:
#     ii = i[1]
#     num = len(ii['charge_type'].unique())
#     if num != 1:
#         print(num)
#         print(ii[['ad_id', 'changeTime', 'charge_type', 'operateType','changeDay']])
    # print(ii[['changeDay','charge_type']])
# opstatus.index = opstatus['statime']
#     opstatus.sort_index()
train['charge_type'] = -1
train['target_type'] = -1
group = train.groupby('ad_id')
new_train = pd.DataFrame()
for g in group:
    ads = g[1]
    aid = ads['ad_id'].values[0]
    op = operate[operate['ad_id'] == aid]
    op.index = op['changeTime']
    op.sort_index()
    targe_type = op['target_type'].values
    charge_type = op['charge_type'].values
    changeDay = op['changeDay'].values
    for i, item in enumerate(changeDay):
        mask = ads['day'] >= item
        print(item)
        ads.loc[mask, 'charge_type'] = targe_type[i]
        ads.loc[mask, 'target_type'] = targe_type[i]
    print(ads)
    new_train = pd.concat([new_train, ads])
new_train.to_csv('add_op_train.csv', index=False)