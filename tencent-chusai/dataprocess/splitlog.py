import pandas as pd
import numpy as np
import time
import gc

name = ['log_0_1999', 'log_2000_3999', 'log_4000_5999','log_6000_7999', 'log_8000_9999', 'log_0_9999', 'log_10000_19999',
        'log_20000_29999', 'log_30000_39999','log_40000_49999',
        'log_50000_59999','log_60000_69999','log_70000_79999','log_80000_89999','log_90000_99999',
        'log_100000_109999','log_110000_119999','log_120000_129999','log_130000_139999',
        'log_140000_149999','log_150000_159999','log_160000_169999','log_170000_179999',
        'log_180000_189999','log_190000_199999','log_200000_209999','log_210000_219999',
        'log_220000_229999','log_230000_239999','log_240000_249999','log_250000_259999',
        'log_260000_269999','log_270000_279999','log_280000_289999','log_290000_299999',
        'log_300000_309999','log_310000_319999','log_320000_329999','log_330000_339999',
        'log_340000_349999','log_350000_359999','log_360000_369999','log_370000_379999',
        'log_380000_389999','log_390000_399999','log_400000_409999','log_410000_419999',
        'log_420000_429999','log_430000_439999','log_440000_449999','log_450000_459999',
        'log_460000_469999','log_470000_479999','log_480000_489999','log_490000_499999',
        'log_500000_509279']
userdata = pd.read_csv('../../user_data.csv')
adoption = pd.read_csv('../../ad_operation.csv')
test = pd.read_csv('../../test_sample.csv')
'''
    将bid变化比较明显的，且不在操作表里，也不在测试集中出现的aid都删掉，
    存储两份，一份是aid是cpc计价的，另一份是非cpc计价的
'''
test_aid = test['ad_id'].unique()
operate_aid = adoption['ad_id'].unique()
for na in name:
    logdata = pd.read_csv('../data/logday2/' + str(na) + '.csv')
    delete = []
    ad = logdata['ad_id'].unique()
    # 遍历aid，将要删除的找出来
    for aid in ad:
        logi = pd.DataFrame()
        logi = logdata[logdata['ad_id'] == aid]
        day = []
        for i in logi['adRequestTime']:
            tt = time.localtime(i)
            day.append(str(tt.tm_mon) + '_' +str(tt.tm_mday))
        logi.loc[ : ,'day'] = day
        sett = logi['day'].unique()
        option = adoption[(adoption['ad_id'] == aid) & (adoption['changeValue'] == '2')]
        num = option.shape[0]
        for day in sett:
            log = logi[logi['day'] == day]
            bid = log['adBid'].unique()
            bidlen = len(bid)
            if((bidlen >= num) & (bidlen > 10)) :
                if((aid not in test_aid) & (aid not in operate_aid)):
                    delete.append(aid)
                    break

    logdata1 = logdata[~(logdata['ad_id'].isin(delete))]
    logdata2 = logdata[logdata['ad_id'].isin(delete)]
    if not logdata1.empty:
        aa = logdata['ad_id'].unique()
        flag = 1
        for i in delete:
            if i in aa:
                flag = 0
                break
            else:
                continue                
        if(flag == 1):
            print(na + "delete sucess")
        if(flag == 0):
            print(na + "delete not sucess")
        user = logdata['user_id'].unique()
        udata = userdata[userdata['user_id'].isin(user)]
        udata.to_csv('../data/user/user_'+ str(na) + '.csv', index = False)
        logdata1.to_csv('../data/logdel/' + str(na) + '.csv', index = False)
        del udata
        del logdata1
        gc.collect()

    if not logdata2.empty:
        user = logdata2['user_id'].unique()
        udata = userdata[userdata['user_id'].isin(user)]
        udata.to_csv('../data/user_uncpc/user_' + str(na) + '.csv', index=False)
        logdata2.to_csv('../data/log_uncpc/' + str(na) + '.csv', index=False)
        del udata
        del logdata2
        gc.collect()
   
