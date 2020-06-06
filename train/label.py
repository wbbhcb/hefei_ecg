#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:36:59 2019

@author: hcb
"""
import tqdm
import pandas as pd
import numpy as np

def read_list(labelfile, label_num):
    filenameslist = []
    filelabelslist = []
    file_age = []
    file_sex = []

    for i in tqdm.tqdm(range(len(labelfile))):
        col = [tmp for tmp in labelfile.columns if tmp not in ['File_name', 'age', 'sex']]
        label = [0 for i in range(label_num)]
        for tmp in col:
            if pd.isnull(labelfile[tmp][i]) == False:
                label[int(labelfile[tmp][i])] = 1
        filenameslist.append(labelfile['File_name'][i])
        if pd.isnull(labelfile['age'][i]) == False:
            file_age.append(labelfile['age'][i])
        else:
            file_age.append(0)
        if pd.isnull(labelfile['sex'][i]) == False:
            file_sex.append(labelfile['sex'][i])
        else:
            file_sex.append(3)

        filelabelslist.append(label)

    return filenameslist, filelabelslist, file_age, file_sex

def get_label():
    label_file = 'hf_round2_train.txt'
    arrythmia = 'lgb/hf_round2_arrythmia.txt'
    label_num = 34
    label = pd.read_csv(label_file, delimiter='\t', header=None, names=['File_name', 'age', 'sex', 0, 1, 2, 3, 4, 5, 6]) # delimiter='\t'
#    label.drop([8], axis=1, inplace=True)
    label_map = pd.read_csv(arrythmia, delimiter='\t', header=None, encoding='utf-8')
#    label_map.drop([1,2,3], axis=1, inplace=True)
    label_dict = dict(zip(label_map[0].unique(),range(label_map[0].nunique())))
    label_map['code'] = label_map[0].map(label_dict)
    sex_map = {'FEMALE':0, 'MALE':1}
    label['sex'] = label['sex'].map(sex_map)
    for i in range(7):
        label[str(i)+'_code'] = label[i].map(label_dict)
    label.drop([0, 1, 2, 3, 4, 5, 6], axis=1, inplace=True)
    filenameslist, filelabelslist, file_age, file_sex = read_list(label, label_num)
    filelabelslist = np.array(filelabelslist)
#    df = pd.DataFrame(filelabelslist)
#    df['file_name'] = filenameslist
    return filenameslist, filelabelslist