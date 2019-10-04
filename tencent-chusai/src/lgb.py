import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

'''
   读入数据，
   1 原始数据：元数据非onehot的类别+onehot展开特征；
   2 单值的出现次数、转化率特征：最大，最小，均值，中位数
   3 多值的最大出现次数、转化率：最大，最小，均值，中位数
   4 交叉特征

'''
fea = [ 'ad_id', 'adBid', 'ad_count_id', 'goods_id', 'goods_type', 'ad_industry_id']
totalTrain = pd.read_csv('../data/TotalTrain_cpc_old.csv')
test = pd.read_csv('../data/test3.csv')
print("spliting train and val data")
y_val = totalTrain[totalTrain['Reqday'] == '03_19']['exposure']
y_train = totalTrain[totalTrain['Reqday'] < '03_19']
y_train = y_train[y_train['Reqday'] > '02_16']['exposure']
print(y_train.shape)

X_train = totalTrain[totalTrain['Reqday'] < '03_19']
X_train = X_train[X_train['Reqday'] > '02_16'][fea]
X_val = totalTrain[totalTrain['Reqday'] == '03_19'][fea]
X_test = test[fea]
num_16 = totalTrain[totalTrain['Reqday'] == '02_16'].shape[0]
print(num_16)
print(X_train.shape)
print(X_val.shape)
print(y_val.shape)

# 原始数据
X_train_meta = pd.read_csv('../data/train/one-hot/X_train_0.csv')
X_train_count = pd.read_csv('../data/train/one-hot/X_train_1.csv')
X_train_labels = pd.read_csv('../data/train/one-hot/X_train_2.csv')
X_train = pd.concat([X_train_meta, X_train_count], axis=1)
X_train = pd.concat([X_train, X_train_labels], axis=1)

X_test_meta = pd.read_csv('../data/train/one-hot/X_test_0.csv')
X_test_count = pd.read_csv('../data/train/one-hot/X_test_1.csv')
X_test_labels = pd.read_csv('../data/train/one-hot/X_test_2.csv')
X_test = pd.concat([X_test_meta, X_test_count], axis=1)
X_test = pd.concat([X_test, X_test_labels], axis=1)
X_test = X_test_meta
print(X_test_labels.shape)
print(X_test.shape)

X_val_meta = pd.read_csv('../data/train/one-hot/X_val_0.csv')
X_val_count = pd.read_csv('../data/train/one-hot/X_val_1.csv')
X_val_labels = pd.read_csv('../data/train/one-hot/X_val_2.csv')
X_val = pd.concat([X_val_meta, X_val_count], axis=1)
X_val = pd.concat([X_val, X_val_labels], axis=1)
print(X_val_labels.shape)
X_val = X_val_meta
print(X_val.shape)


# 训练lgb
gbm = lgb.LGBMRegressor(objective='mape',
                        subsample=1,
                        min_child_weight=0.5,
                        colsample_bytree=1,  # 在每棵树训练之前选择一部分特征来训练。
                        num_leaves=30,  # 默认是31
                        max_depth=5,  # 默认是-1，表示没有限制
                        learning_rate=0.005,  # 默认是0.1
                        n_estimators=100000,
                        lambda_l2=5,  #正则化系数，越大则约不容易过拟合
                        train_metric=True,
                        )

gbm.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['X_train', 'val'],
        eval_metric='mape',
        early_stopping_rounds=100,
        )


model = gbm.booster_
predict = model.predict(X_test)
sub = []
for i in predict:
    if i > 0:
        sub.append(i)
    else:
        sub.append(0)
mean_bid = test['adBid'].groupby(test['ad_id']).mean().reset_index()
mean_bid = mean_bid.rename(columns={'adBid': 'mean_bid'})
test2 = pd.merge(test, mean_bid, on='ad_id', how='left')
sub = sub + (test2['adBid'] - test2['mean_bid'])/test2['mean_bid']

submit = pd.DataFrame()
submit['id'] = range(1, 20291)
submit['ex'] = sub
submit.to_csv('submission_lgb.csv', index=False,header=None)

importance_dict = {}
for i, col_name in enumerate(X_train.columns):
    importance_dict[col_name] = gbm.feature_importances_[i]
sort_dict = sorted(dict.items(importance_dict), key=lambda x: x[1], reverse=True)
print(sort_dict)
keys = []
importance = []
for k in sort_dict:
    keys.append(k[0])
    importance.append(k[1])
fea_imp = pd.DataFrame()
fea_imp['feature'] = keys
fea_imp['importance'] = importance
fea_imp.to_csv('feature_importance_total.csv', index=False)




