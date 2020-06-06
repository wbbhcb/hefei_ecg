#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:52:28 2019

@author: hcb
"""

import pandas as pd
import os
import numpy as np
import pickle
import tqdm
from config_lgb import config

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
fold = [0, 1, 2, 3, 4]

sub_file = config.sub_file
arrythmia = 'hf_round2_arrythmia.txt'
base_path1 = 'DL_feature'
label_num = 34
use_age_sex = 1

sub = pd.read_csv(sub_file, header=None, delimiter='\t', names=['filename', 'age', 'sex', 1, 2, 3, 4])
sub.drop([1, 2, 3, 4], axis=1, inplace=True)
label_map = pd.read_csv(arrythmia, delimiter='\t', header=None, encoding='utf-8', engine='python')
# label_map.drop([1, 2, 3], axis=1, inplace=True)
label_dict = dict(zip(label_map[0].unique(), range(label_map[0].nunique())))
label_map['code'] = label_map[0].map(label_dict)
sex_map = {'FEMALE': 0, 'MALE': 1}
sub['sex_new'] = sub['sex'].map(sex_map)
# df = pd.DataFrame()
result = []
add_ml_feature = 1
feature_len = 512
feature_len2 = 1024

dl_name1 = 'myecgnet'
#dl_name2 = 'resnet34'

if add_ml_feature == 1:
    data2 = pd.read_csv('train_ml.csv', header=0, engine='python')
pred = np.zeros((len(sub), label_num))
prob_dl = np.zeros((len(sub), label_num))

for i in fold:
    # read data
    print('-----------' + str(i) + '---------------')
    df = sub.copy()
    data = pd.read_csv(os.path.join(base_path1, 'prob_'+dl_name1 + '_' + 
                                    str(i)+'.csv'), header=0, engine='python')
    col1 = [str(i) for i in range(34)] + ['filename']
    data = data[col1]
    col1 = ['prob1_'+str(i) for i in range(34)] + ['filename']
    data.columns = col1
    col1 = col1 = ['prob1_'+str(i) for i in range(34) ]
#if i not in [0, 2, 6, 7, 8, 14, 15, 16, 19, 21, 23, 25, 32, 33]    
    
    data_2 = pd.read_csv(os.path.join(base_path1, dl_name1 + '_' + 
                                      str(i) + '_test.csv'), header=0, engine='python')
    col2 = [str(i) for i in range(feature_len)] + ['filename']
    data_2 = data_2[col2]
    col2 = ['df1_'+str(i) for i in range(feature_len)] + ['filename']
    data_2.columns = col2
    col2 = ['df1_'+str(i) for i in range(feature_len)]
    
    data = data.merge(data_2, 'left', on='filename')
    
    
#    data_2 = pd.read_csv(os.path.join(base_path1, 'prob_'+dl_name2 + '_' + 
#                                    str(i)+'.csv'), header=0, engine='python')
#    col3 = [str(i) for i in range(34)] + ['filename']
#    data_2 = data_2[col3]
#    col3 = ['prob2_'+str(i) for i in range(34)] + ['filename']
#    data_2.columns = col3
#    col3 = ['prob2_'+str(i) for i in range(34)]
#    
#    data = data.merge(data_2, 'left', on='filename')
#    
#    data_2 = pd.read_csv(os.path.join(base_path1, dl_name2 + '_' + 
#                                      str(i) + '_test.csv'), header=0, engine='python')
#    col4 = [str(i) for i in range(feature_len2)] + ['filename']
#    data_2 = data_2[col4]
#    col4 = ['df2_'+str(i) for i in range(feature_len2)] + ['filename']
#    data_2.columns = col4
#    col4 = ['df2_'+str(i) for i in range(feature_len2)]
#    
#    data = data.merge(data_2, 'left', on='filename')
    
    
    if add_ml_feature == 1:
        data = data.merge(data2[[str(col) for col in range(1336)] + ['filename']],
                          'left', on='filename')

    df = df.merge(data, 'left', on='filename')
    
    prob_dl += sigmoid(df[['prob1_'+str(i) for i in range(34)]].values)
    
    col = col1 + col2 + [str(ii) for ii in range(1336)] + ['age', 'sex_new']
    
#    col_idx = [i for i in range(len(col1))] + [i+len(col1)+len(col2) for i in range(1338)]
    
    
    val_data = df[col].values
    print(val_data.shape)
    pred_fold = np.zeros((len(sub), label_num))
    for j in range(label_num):
        print('folds: %d, label: %d' % (i, j))
        if use_age_sex == 0:
            save_base_path = os.path.join('model', str(i))
        else:
            save_base_path = os.path.join('model', str(i))
        # print(save_base_path)
        save_path = os.path.join(save_base_path, 'model_lgb' + str(j) + '.pickle')
        with open(save_path, 'rb+') as f:
            clf = pickle.load(f)
            
#        if j in [17, 18, 20, 30]:
#            pred_fold[:, j] = clf.predict(val_data[:,len(col1):], num_iteration=clf.best_iteration)
#        elif j in [27, 28, 31]:
#            pred_fold[:, j] = clf.predict(val_data[:, col_idx], num_iteration=clf.best_iteration)
#        else:
        pred_fold[:, j] = clf.predict(val_data, num_iteration=clf.best_iteration)
            
#        pred_fold[:, j] = clf.predict(val_data, num_iteration=clf.best_iteration)
    pred = pred + pred_fold
pred = pred / len(fold)


#prob_dl = prob_dl / len(fold)
#for i in [1, 10, 12, 13]:
#    pred[:, i] = prob_dl[:, i]
#pred = pred * 0.8 + prob_dl / len(fold) * 0.2

pred = np.round(pred)
print(np.sum(pred, axis=0))
# csvName = "result.txt"
f = open(config.result_path, "w+", encoding='utf-8')
for i in tqdm.tqdm(range(len(sub))):
    new_context = sub['filename'][i] + '\t'
    if pd.isnull(sub['age'][i]) or pd.isna(sub['age'][i]):
        new_context += ' ' + '\t'
    else:
        new_context += str(int(sub['age'][i])) + '\t'

    if pd.isnull(sub['sex'][i]) or pd.isna(sub['sex'][i]):
        new_context += ' '
    else:
        new_context += sub['sex'][i]
    for j in range(label_num):
        if pred[i, j] == 1:
            new_context = new_context + '\t' + list(label_map[0][label_map['code'] == j])[0]
    new_context += '\n'
    f.write(new_context)
f.close()
