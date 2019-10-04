# -*- coding: utf-8 -*-
"""
@file:test_feature1.py
@time:2019/6/1 20:06
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import time
tt = time.time()
fea = pd.DataFrame()
test_bid = pd.read_csv('../usingData/test/test_bid.csv')
request = pd.read_csv('../usingData/test/Request_list.csv')
log = pd.read_csv('../usingData/test/test_log.csv')
print(test_bid.shape)
print(request.columns)
print(log.columns)
# 先把比较简单的，request长度找出来
fea['ad_id'] = request['ad_id'].values
ad_id = request['ad_id'].values
request_list = request['RequestList'].values
request_len = []
for ad in request_list:
    ads = ad.split("|")
    lens = len(ads)
    request_len.append(lens)
fea['request_nums'] = request_len

print(time.localtime(tt))
'''
请求的uid的个数
找出对应的人群的，每一个广告都有一个dataframe，然后将那个log文件和它merge起来，然后统计uid的个数就好了呗
把log里面只取出来uid和对应的request id和positionid，然后将其用逗号连接，变成一个字符串
'''

log_uid = log['uid'].values
log_reqid = log['RequestId'].values
log_posiId = log['PositionId'].values
request = []
for i, k in enumerate(log_uid):
    s = str(log_reqid[i]) + ',' + str(log_posiId[i])
    request.append(s)
log_new = pd.DataFrame()
log_new['uid'] = log_uid
log_new['request'] = np.array(request)
uid_len = []
k = 0
# 找出每个aid对应的request集合，然后作为一个dataframe，和上面的进行merge
for request_l in request_list:
    tt = time.time()
    request_i = request_l.split('|')
    request_new = pd.DataFrame()
    request_new['request'] = request_i
    # print(request_new.shape)
    request_new = pd.merge(request_new, log_new, on='request', how='left')
    # print(request_new.shape)
    uid_len.append(request_new['uid'].unique().shape[0])
    print(k)
    k += 1
fea['uid_nums'] = uid_len
fea.to_csv('../usingData/feature/uid_req_test.csv', index=False)
