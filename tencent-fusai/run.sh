# 先生成基础特征的train集合
python dataprocess/make_metafea_train.py

# 再利用基础特征的train，生成对应的feature文件
python src/chusai_feature_preparing.py

python src/test_feature1.py
python src/train_feature1.py

python src/test_feature2.py
python src/train_feature2.py

python src/test_feature3.py
python src/train_feature3.py

python src/compete_pctr_yes_test.py
python src/compete_pctr_yes_train.py

python src/make_cross_rate_feature.py
python src/make_new_ad_cross_rate.py
python src/make_old_ad_cross_rate.py

python src/make_option_type_feature.py

python src/meta_pctr.py
python src/request_nums_position_part_test.py

# 之后这些特征文件，要么在lgb.py中直接作为特征被调用，要么是作为rule中的应用。
python modelB/lgb.py
