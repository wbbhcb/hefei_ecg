# -*- coding: utf-8 -*-

import os


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    # root = r'../'

    sub_file = '../hf_round2_train.txt'
    feature = '_train.csv'
    root = r'/home/hcb/桌面/tcdata'
    arrythmia_file = 'hf_round2_arrythmia.txt'
    file_path = os.path.join(root, 'hf_round2_train')
    #label的类别数
    num_classes = 34
    #最大训练多少个epoch
    sub_dir = '../../'
    result_path = os.path.join(sub_dir, 'result.txt')
    save_name = 'train_ml.csv'
config = Config()
