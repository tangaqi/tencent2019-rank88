import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from time import time
import math
import xgboost as xgb
from sklearn.metrics import *
import pandas as pd
from utils import *
part = int(sys.argv[1])
# 预定义各类list
one_fea_statistic = ['ad_count_id', 'goods_id', 'goods_type' ,'ad_size', 
               'ad_industry_id','ad_id','puttingTime']
vector_fea_statistic = ['area','behavior','age', 'gender',
              'education', 'consuptionAbility', 'device', 'connectionType','work',
              'status']
dynamic_dict = { 'age':93+2, 'gender':5+1,
              'education':9+1, 'consuptionAbility':4+1, 'device':5+1, 'connectionType':6+1,
                'work':8+1,'status':20+1,'area':13647,'behavior':32632}
# 'adBid' 是一个int类型的连续特征
# 原始onehot特征，可以直接输入到xgb中的
one_hot_fea = ['goods_type' ,'ad_size', 'ad_industry_id','puttingTime']
vector_fea = ['age', 'gender','education', 'consuptionAbility', 'device', 
              'connectionType','work','status']

date = ['2_16','2_17','2_18' ,'2_19' ,'2_20' ,'2_21' ,'2_22' ,'2_23' ,'2_24', '2_25' ,'2_26',
 '2_27', '2_28', '3_1', '3_2' ,'3_3', '3_4' , '3_5' ,'3_6' ,'3_7' ,'3_8' ,'3_9',
             '3_10', '3_11', '3_12' ,'3_13','3_14' ,'3_15', '3_16' ,'3_17', '3_18' ,'3_19' ]
# 将all特征读入
age_all, age_dict = read_allfea('../data/usefe/age.txt')
gender_all, gender_dict = read_allfea('../data/usefe/gender.txt')
education_all, education_dict = read_allfea('../data/usefe/education.txt')
consuptionAbility_all, consuptionAbility_dict = read_allfea('../data/usefe/consupall.txt')
device_all, device_dict = read_allfea('../data/usefe/deviceall.txt')
connectionType_all, connectionType_dict = read_allfea('../data/usefe/connectionall.txt')
work_all, work_dict = read_allfea('../data/usefe/work.txt')
status_all, status_dict = read_allfea('../data/usefe/statusall.txt')
feature_dict = {'age':age_dict,'gender':gender_dict,'education':education_dict,
            'consuptionAbility':consuptionAbility_dict,'device':device_dict,
            'connectionType':connectionType_dict,'work':work_dict,'status':status_dict}

print("reading train data")
totalTrain = pd.read_csv('../data/TotalTrain4.csv')
print("spliting train and val data")
val = totalTrain[totalTrain['Reqday'] == '3_19']
train = totalTrain[totalTrain['Reqday'] < '3_19']
y_train = train['exposure']
y_val = val['exposure']
print(train.shape)
print(val.shape)
print(y_val.shape)
print(y_train.shape)
print("reading test data")
test = pd.read_csv('../data/test3.csv')
num1 = train.shape[0]
num2 = val.shape[0] + num1
num3 = test.shape[0] + num2

print(y_train.shape)
y_train.to_csv('../data/Amiddle-fea/one-hot/y_train.csv', index=False)
print(y_val.shape)
y_val.to_csv('../data/Amiddle-fea/one-hot/y_val.csv', index=False)

        
if part == 0:
    '''
    将原始特征读入，单值先进行index编码，然后将部分维度较小的展开，多值特征直接返回的是一个muti-hot的编码
    '''
    print('one_hot_fea preparing')
    train_use = pd.DataFrame()
    val_use = pd.DataFrame()
    test_use = pd.DataFrame()
    for fea in one_hot_fea:
        print(fea + 'preparing')
        train_res, val_res, predict_res, f_dict = one_hot_feature_process(train[fea].values, val[fea].values, test[fea].values, 1)
        write_dict('../data/Amiddle-fea/dict/' + fea + '.csv',f_dict)    
        train_use[fea] = train_res
        val_use[fea] = val_res
        test_use[fea] = predict_res
    print('one hot meta feature train_use shape:', train_use.shape)
    print('val_use shape:', val_use.shape)
    print('test_use shape:', test_use.shape)
    Train_rate = pd.DataFrame()
    Val_rate = pd.DataFrame()
    Test_rate = pd.DataFrame()
    for fea in vector_fea:
        print(fea + 'preparing')
        max_len = dynamic_dict[fea]
        fea_dict = feature_dict[fea]
        begin_index = len(fea_dict) + 1
        train_res, val_res, predict_res, f_dict,train_res2, val_res2, predict_res2, train_rate, val_rate, test_rate = vector_feature_process(train[fea].values, val[fea].values, test[fea].values, begin_index, max_len, fea_dict)
        write_dict('../data/Amiddle-fea/dict/' + fea + '.csv',fea_dict)
        for i in range(train_res2.shape[1]):
            col = fea + str(i)
            train_use[col] = train_res2[:, i]
        for i in range(val_res2.shape[1]):
            col = fea + str(i)
            val_use[col] = val_res2[:, i]
        for i in range(predict_res2.shape[1]):
            col = fea + str(i)
            test_use[col] = predict_res2[:, i]
        col = fea + '_len'
        Train_rate[col] = train_rate
        Test_rate[col] = test_rate
        Val_rate[col] = val_rate
    
    print('+ muti hot  train_use shape:', train_use.shape)
    print('val_use shape:', val_use.shape)
    print('test_use shape:', test_use.shape)
    # 人群定向率计算
    train_peo = np.array([1] * num1)
    val_peo = np.array([1] * val.shape[0])
    test_peo = np.array([1] * test.shape[0])
    for i in Train_rate.columns:
        train_peo = train_peo * Train_rate[i].values
        test_peo = test_peo * Test_rate[i].values
        val_peo = val_peo * Val_rate[i].values
    train_use['peo_rate'] = train_peo
    val_use['peo_rate'] = val_peo
    test_use['peo_rate'] = test_peo
    
    train_use = pd.concat([train_use, Train_rate], axis = 1)
    val_use = pd.concat([val_use, Val_rate], axis = 1)
    test_use = pd.concat([test_use, Test_rate], axis = 1)
    print('+ len fea train_use shape:', train_use.shape)
    print('val_use shape:', val_use.shape)
    print('test_use shape:', test_use.shape)
    one_hot_data = pd.concat([train_use, val_use])
    one_hot_data = pd.concat([one_hot_data, test_use])
    for fea in one_hot_fea:
        # 先进行onehot编码，将所有的都拼接在一起，这样才能保证一致性，然后按照大小将其分给训练集验证集和测试集
        onehot = pd.get_dummies(one_hot_data[fea], prefix = fea)
        print(fea + 'one_hot ing')
        train_use.drop([fea], axis = 1, inplace = True)
        train_use = pd.concat([train_use, onehot[:num1]], axis = 1)
        print('train ', train_use.shape)
        val_use.drop([fea], axis = 1, inplace = True)
        val_use = pd.concat([val_use, onehot[num1:num2]], axis = 1)
        print('val ', val_use.shape)
        test_use.drop([fea], axis = 1, inplace = True)
        test_use = pd.concat([test_use, onehot[num2:num3]], axis = 1)
        print('test ', test_use.shape)
        
    print('save X_train_use shape:', train_use.shape)
    train_use.to_csv('../data/Amiddle-fea/one-hot/X_train_0.csv', index = False)
    print('save X_test_use shape:', test_use.shape)
    val_use.to_csv('../data/Amiddle-fea/one-hot/X_val_0.csv', index = False)
    print('save X_val_use shape:', val_use.shape)
    test_use.to_csv('../data/Amiddle-fea/one-hot/X_test_0.csv', index = False)

if part == 1:
    '''
    返回的是统计特征，也就是各个特征出现的频次。对于多值特征，返回的是平均出现次数和出现的最大的次数
    '''    
    # 将所有的all值替换掉
    test.loc[test['age'] == 'all', 'age'] = age_all
    test.loc[test['gender'] == 'all', 'gender'] = gender_all
    test.loc[test['education'] == 'all', 'education'] = education_all
    test.loc[test['consuptionAbility'] == 'all', 'consuptionAbility'] = consuptionAbility_all
    test.loc[test['device'] == 'all', 'device'] = device_all
    test.loc[test['connectionType'] == 'all', 'connectionType'] = connectionType_all
    test.loc[test['work'] == 'all', 'work'] = work_all
    test.loc[test['status'] == 'all', 'status'] = status_all
    
    val.loc[val['age'] == 'all', 'age'] = age_all
    val.loc[val['gender'] == 'all', 'gender'] = gender_all
    val.loc[val['education'] == 'all', 'education'] = education_all
    val.loc[val['consuptionAbility'] == 'all', 'consuptionAbility'] = consuptionAbility_all
    val.loc[val['device'] == 'all', 'device'] = device_all
    val.loc[val['connectionType'] == 'all', 'connectionType'] = connectionType_all
    val.loc[val['work'] == 'all', 'work'] = work_all
    val.loc[val['status'] == 'all', 'status'] = status_all
    
    train.loc[train['age'] == 'all', 'age'] = age_all
    train.loc[train['gender'] == 'all', 'gender'] = gender_all
    train.loc[train['education'] == 'all', 'education'] = education_all
    train.loc[train['consuptionAbility'] == 'all', 'consuptionAbility'] = consuptionAbility_all
    train.loc[train['device'] == 'all', 'device'] = device_all
    train.loc[train['connectionType'] == 'all', 'connectionType'] = connectionType_all
    train.loc[train['work'] == 'all', 'work'] = work_all
    train.loc[train['status'] == 'all', 'status'] = status_all
    
    print('statistics_fea preparing')
    train_use = pd.DataFrame()
    val_use = pd.DataFrame()
    test_use = pd.DataFrame()
    train_use['adBid'] = train['adBid'].values
    val_use['adBid'] = val['adBid'].values
    test_use['adBid'] = test['adBid'].values
    
    for fea in one_fea_statistic:
        print(fea + ' statistic preparing')
        col = fea + 'count'
        train_res, val_res, test_res = count_one_feature_times(train, val, test[fea].values, fea, date)
        train_use[col] = train_res
        val_use[col] = val_res
        test_use[col] = test_res        
        
    for fea in vector_fea:
        print(fea + ' statistic preparing')
        col = fea + 'count'
        flag = 0
        train_res, val_res, test_res = count_vector_feature_times(train, val, test[fea].values, fea, date, flag)
        # 这里返回的是次数的最大值和均值
        for i in range(2):
            if i == 0:
                col = col + '_' + 'max'
            else:
                col = col + '_' + 'average'
            train_use[col] = train_res[:, i]
            val_use[col] = val_res[:, i]
            print(test.shape)
            print(test_res.shape)
            test_use[col] = test_res[:, i]    
    # 这里虽然不需要进行onehot编码，但是需要对数值特征进行归一化操作
    # num_1 = train[train['Reqday'] == '2_16'].shape[0]
    for c in train_use.columns:
        train_use.loc[:, c] = StandardScaler().fit(train_use.loc[:, [c]]).transform(train_use.loc[:, [c]])
    for c in val_use.columns:
        val_use.loc[:, c] = StandardScaler().fit(val_use.loc[:, [c]]).transform(val_use.loc[:, [c]])
    for c in test_use.columns:
        test_use.loc[:, c] = StandardScaler().fit(test_use.loc[:, [c]]).transform(test_use.loc[:, [c]])
    train_use.to_csv('../data/Amiddle-fea/one-hot/X_train_1.csv', index = False)
    val_use.to_csv('../data/Amiddle-fea/one-hot/X_val_1.csv', index = False)
    test_use.to_csv('../data/Amiddle-fea/one-hot/X_test_1.csv', index = False)


if part == 2:
    # 对转化率特征进行计算
    train_use = pd.DataFrame()
    val_use = pd.DataFrame()
    test_use = pd.DataFrame()
    col_name1 = ['max_exposure', 'min_exposure', 'mean_exposure', 'median_exposure']
    col_name2 = ['max_bid', 'min_bid', 'mean_bid', 'median_bid']
    for fea in one_hot_fea:
        print(fea + "label feature preparing")
        train_res, val_res, test_res,train_res2, val_res2, test_res2 = one_feature_exposure(train, val,                                                                                       test[fea].values, fea, date)
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
    for fea in vector_fea:
        print(fea + "label feature preparing")
        train_res, val_res, test_res, train_res2, val_res2, test_res2 = \
            vector_feature_exposure(train, val, test[fea].values, fea, date)
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
    
    # 对数据进行归一化处理
    # num_1 = train[train['Reqday'] == '2_16'].shape[0]
    for c in train_use.columns:
        train_use.loc[:, c] = StandardScaler().fit(train_use.loc[:, [c]]).transform(train_use.loc[:, [c]])
    for c in val_use.columns:
        val_use.loc[:, c] = StandardScaler().fit(val_use.loc[:, [c]]).transform(val_use.loc[:, [c]])
    for c in test_use.columns:
        test_use.loc[:, c] = StandardScaler().fit(test_use.loc[:, [c]]).transform(test_use.loc[:, [c]])

    train_use.to_csv('../data/Amiddle-fea/one-hot/X_train_3.csv', index=False)
    val_use.to_csv('../data/Amiddle-fea/one-hot/X_val_3.csv', index=False)
    test_use.to_csv('../data/Amiddle-fea/one-hot/X_test_3.csv', index=False)
    
    
    
    



