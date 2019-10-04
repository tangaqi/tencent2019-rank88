# -*- coding: utf-8 -*-
"""
@file:lgb.py
@time:2019/6/7 18:00
@author:Tangj
@software:Pycharm
@Desc
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

cross_train = ['ad_id', 'day','goods_id|ad_size_meancount', 'ad_count_id|ad_size_meancount', 'ad_industry_id|goods_type']
cross_test = ['ad_id', 'bid', 'goods_id|ad_size_meancount', 'ad_count_id|ad_size_meancount', 'ad_industry_id|goods_type']
fea = ['ad_id', 'day', 'createTime', 'ad_count_id', 'goods_id', 'goods_type', 'ad_industry_id', 'ad_size']
fea_test = ['ad_id', 'bid', 'ad_count_id', 'goods_id', 'goods_type', 'ad_industry_id', 'createTime', 'ad_size']
totalTrain = pd.read_csv('../usingData/train/metafea_train.csv')
new_ex = pd.read_csv('../usingData/train/exposure_new.csv')
totalTrain = totalTrain.rename(columns={'ex': 'meta_ex'})
totalTrain = pd.merge(totalTrain, new_ex, on=['ad_id', 'day'], how='left')
totalTrain = totalTrain.fillna(0.0)
print(totalTrain[totalTrain['ex'] == 0].shape)
Y_train = totalTrain['ex']
train = totalTrain
testdata = pd.read_table('../metaData/metaTest/Btest_sample_bid.out', header=None)
test = pd.DataFrame()
test['ad_id'] = testdata[1]
test['bid'] = testdata[4]
static = pd.read_csv('../usingData/train/ad_static.csv')
test = pd.merge(test, static, on='ad_id', how='left')

y_train = Y_train.values
X_train = train[fea]
X_test = test[fea_test]
# 添加count特征
test_count = pd.read_csv('../usingData/feature/count_fea_test.csv')
train_count = pd.read_csv('../usingData/feature/count_fea_train.csv')
print(test_count.columns)
print(train.columns)
X_test = pd.merge(X_test, test_count, on=['ad_id', 'bid'], how='left')
X_train = pd.merge(X_train, train_count, on=['ad_id', 'day'], how='left')

# 添加cross特征
train_cross = pd.read_csv('../usingData/feature/cross_fea_train.csv')
test_cross = pd.read_csv('../usingData/feature/cross_fea_test.csv')
X_test = pd.merge(X_test, test_cross, on=['ad_id', 'bid'], how='left')
X_train = pd.merge(X_train, train_cross, on=['ad_id', 'day'], how='left')
print(X_train.shape)
print(X_test.shape)

# 这里的aid mean rate是历史平移特征，平移了两天的rate特征
uid_train = pd.read_csv('../usingData/feature/aid_rate_train.csv')[['ad_id_mean_rate', 'ad_id', 'day']]
uid_test = pd.read_csv('../usingData/feature/aid_rate_test.csv')[['ad_id_mean_rate']]
print(uid_train.columns)
X_test = pd.concat([X_test, uid_test], axis=1)
X_train = pd.merge(X_train, uid_train, on=['ad_id', 'day'], how='left')

# mean_exposure里面其实只用到了对应的uid nums 和 request  nums，
uid_train = pd.read_csv('../usingData/feature/mean_exposure_train.csv')
uid_test = pd.read_csv('../usingData/feature/mean_exposure_test.csv')
print(uid_train.columns)
X_test = pd.merge(X_test, uid_test, on=['ad_id'], how='left')
X_train = pd.merge(X_train, uid_train, on=['ad_id', 'day'], how='left')

X_train['cal_ex'] = X_train['ad_id_mean_rate'].values * X_train['request_nums'].values
X_test['cal_ex'] = X_test['ad_id_mean_rate'].values * X_test['request_nums'].values
X_test.drop(['bid'], axis=1, inplace=True)

history_train = pd.read_csv('../usingData/feature/history_fea_label_train.csv')[['ad_id', 'day', 'ad_id_mean_exposure']]
history_test = pd.read_csv('../usingData/feature/history_fea_label_test.csv')[['ad_id', 'ad_id_mean_exposure']]
X_test = pd.merge(X_test, history_test, on=['ad_id'], how='left')
X_train = pd.merge(X_train, history_train, on=['ad_id', 'day'], how='left')
# print(X_train['ad_id_mean_exposure'] - X_train['cal_ex'])

compete_train = pd.read_csv('../usingData/feature/compete_rate_train.csv')
compete_test = pd.read_csv('../usingData/feature/compete_rate_test.csv')
print(compete_train.columns)
print(compete_test.columns)
X_test = pd.merge(X_test, compete_test, on=['ad_id'], how='left')
X_train = pd.merge(X_train, compete_train, on=['ad_id', 'day'], how='left')
print(y_train.shape)
print(X_train.columns)
print(X_test.columns)

compete_train = pd.read_csv('../usingData/feature/cross_fea_rate_train.csv')
compete_test = pd.read_csv('../usingData/feature/cross_fea_rate_test.csv')
compete_test = pd.read_csv('../usingData/feature/cross_fea_rate_test.csv')
compete_train['rate1'] = 0.01*compete_train['goods_id|ad_size_meancount_mean_rate'].values + 0.99*compete_train['goods_id|ad_size_meancount_min_rate'].values
compete_test['rate1'] = 0.01*compete_test['goods_id|ad_size_meancount_mean_rate'].values + 0.99*compete_test['goods_id|ad_size_meancount_min_rate'].values
compete_train['rate2'] = 0.01*compete_train['ad_industry_id|goods_type_mean_rate'].values + 0.99*compete_train['ad_industry_id|goods_type_min_rate'].values
compete_test['rate2'] = 0.01*compete_test['ad_industry_id|goods_type_mean_rate'].values + 0.99*compete_test['ad_industry_id|goods_type_min_rate'].values
compete_train['rate3'] = 0.01*compete_train['ad_count_id|goods_type_mean_rate'].values + 0.99*compete_train['ad_count_id|goods_type_min_rate'].values
compete_test['rate3'] = 0.01*compete_test['ad_count_id|goods_type_mean_rate'].values + 0.99*compete_test['ad_count_id|goods_type_min_rate'].values
compete_train['rate4'] = 0.01*compete_train['ad_industry_id|ad_count_idcount_mean_rate'].values + 0.99*compete_train['ad_industry_id|ad_count_idcount_min_rate'].values
compete_test['rate4'] = 0.01*compete_test['ad_industry_id|ad_count_idcount_mean_rate'].values + 0.99*compete_test['ad_industry_id|ad_count_idcount_min_rate'].values
compete_train['rate5'] = 0.01*compete_train['ad_industry_id|ad_count_idcount_mean_rate'].values + 0.99*compete_train['ad_industry_id|ad_count_idcount_min_rate'].values
compete_test['rate5'] = 0.01*compete_test['ad_industry_id|ad_count_idcount_mean_rate'].values + 0.99*compete_test['ad_industry_id|ad_count_idcount_min_rate'].values

compete_train['rate6'] = 0.01*compete_train['ad_count_id|ad_size_mean_mean_rate'].values + 0.99*compete_train['ad_count_id|ad_size_mean_min_rate'].values
compete_test['rate6'] = 0.01*compete_test['ad_count_id|ad_size_mean_mean_rate'].values + 0.99*compete_test['ad_count_id|ad_size_mean_min_rate'].values
compete_train['rate7'] = 0.01*compete_train['goods_id|goods_typecount_mean_rate'].values + 0.99*compete_train['ad_industry_id|goods_type_min_rate'].values
compete_test['rate7'] = 0.01*compete_test['goods_id|goods_typecount_mean_rate'].values + 0.99*compete_test['goods_id|goods_typecount_min_rate'].values
compete_train['rate8'] = 0.01*compete_train['ad_industry_id|ad_count_id|goods_typecount_mean_rate'].values + 0.99*compete_train['ad_industry_id|ad_count_id|goods_typecount_min_rate'].values
compete_test['rate8'] = 0.01*compete_test['ad_industry_id|ad_count_id|goods_typecount_mean_rate'].values + 0.99*compete_test['ad_industry_id|ad_count_id|goods_typecount_min_rate'].values
compete_train['rate9'] = 0.01*compete_train['ad_industry_id|ad_count_id|ad_size_meancount_mean_rate'].values + 0.99*compete_train['ad_industry_id|ad_count_id|ad_size_meancount_min_rate'].values
compete_test['rate9'] = 0.01*compete_test['ad_industry_id|ad_count_id|ad_size_meancount_mean_rate'].values + 0.99*compete_test['ad_industry_id|ad_count_id|ad_size_meancount_min_rate'].values
compete_train['rate10'] = 0.01*compete_train['ad_industry_id|goods_type|ad_size_meancount_mean_rate'].values + 0.99*compete_train['ad_industry_id|goods_type|ad_size_meancount_min_rate'].values
compete_test['rate10'] = 0.01*compete_test['ad_industry_id|goods_type|ad_size_meancount_mean_rate'].values + 0.99*compete_test['ad_industry_id|goods_type|ad_size_meancount_min_rate'].values

X_test = pd.concat([X_test, compete_test[['rate1', 'rate2', 'rate3', 'rate4', 'rate5']]], axis=1)
X_train = pd.merge(X_train, compete_train[['rate1', 'rate2', 'rate3', 'rate4', 'rate5', 'ad_id', 'day']], on=['ad_id', 'day'], how='left')
print(y_train.shape)
print(X_train.columns)
print(X_test.columns)

pctr_train = pd.read_csv('../usingData/new/history_pctr_train.csv')
pctr_test = pd.read_csv('../usingData/new/history_pctr_test.csv')
print(pctr_train.columns)
print(pctr_test.columns)
X_test = pd.merge(X_test, pctr_test, on=['ad_id'], how='left')
X_train = pd.merge(X_train, pctr_train, on=['ad_id', 'day'], how='left')

X_train.drop(['ad_id', 'day', 'mean_exposure', 'rate_mean', 'ad_id_bid', 'ad_id_quality_ecpm'], axis=1, inplace=True)
X_test.drop(['ad_id', 'mean_exposure', 'rate_mean', 'ad_id_bid', 'ad_id_quality_ecpm'], axis=1, inplace=True)
print(y_train.shape)
print(X_train.shape)
print(X_test.shape)
print(X_train.columns)
print(X_test.columns)

# 训练lgb
model = lgb.LGBMRegressor(objective='mape',
                        subsample=1,
                        bagging_freq= 2,
                        bagging_fraction= 0.5,
                        # min_child_weight=0.5,
                        colsample_bytree=0.9,  # 在每棵树训练之前选择一部分特征来训练。
                        num_leaves=100,  # 默认是31
                        max_depth=7,  # 默认是-1，表示没有限制
                        learning_rate=0.01,  # 默认是0.1
                        n_estimators=20000,
                        lambda_l2=8,  #正则化系数，越大则约不容易过拟合
                        lambda_l1=8,
                        train_metric=True,
                        random_state=2019
                        )

folds = StratifiedKFold(n_splits=5)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):
    trn_x, trn_y = X_train.iloc[trn_], y_train[trn_]
    val_x, val_y = X_train.iloc[val_], y_train[val_]

    model.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=300, verbose=100)
    oof_preds[val_] += model.predict(val_x, num_iteration=model.best_iteration_)
    sub_preds += model.predict(X_test, num_iteration=model.best_iteration_) /folds.n_splits

sum = 0
for i, k in enumerate(oof_preds):
    sum += 2*abs(oof_preds[i] - y_train[i])/(max(oof_preds[i],1) + y_train[i])
sum = sum/X_train.shape[0]
print("total smape:", sum)
testdata = pd.read_csv('../usingData/test/test_bid.csv')
test_mask = pd.read_csv('../usingData/feature/uid_req_test.csv')
rate_mask = pd.read_csv('../usingData/feature/rate_expose.csv')
rate_mask1 = rate_mask[['ad_id', 'rate_mean', 'rate_explose13']]
test_mask = pd.merge(testdata, rate_mask1, on='ad_id', how='left')
test_mask = test_mask.fillna(-100)
test_mask['flag'] = test_mask['rate_mean']
mask0 = test_mask['rate_mean'] == -100

submit = pd.DataFrame()
submit['id'] = range(1, test.shape[0] + 1)
submit['ex'] = sub_preds + test['bid']/1000000
print(submit[mask0].describe())
print(submit[~mask0].describe())
print(submit.describe())
submit.to_csv('submission_modelB1.csv', index=False, header=None)