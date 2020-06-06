# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
'''
import os, torch
import numpy as np
from config2 import config
import pandas as pd
from sklearn.model_selection import KFold

# 保证每次划分数据一致
np.random.seed(41)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def split_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)


def my_split_data(file2idx, file_name, fold, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    print(fold)
    my_file = pd.read_csv('mylabel.csv')
    my_file = list(my_file['File_name'])
#    all_idx = [i for i in range(len(file_name))]
    val_idx = [i for i in range(fold, len(file_name), 5)]
    trn_idx = [i for i in range(len(file_name)) if i not in val_idx]
    train = []
    val = []
    for i in range(len(val_idx)):
        if file_name[val_idx[i]] in my_file:
            val.append(file_name[val_idx[i]])
    for i in range(len(trn_idx)):
        if file_name[trn_idx[i]] in my_file:
            train.append(file_name[trn_idx[i]])
#    train = data.difference(val)
    print(len(val), len(train))
    return list(train), list(val)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    file_name = []
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
        file_name.append(id)
    return file2index, file_name


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def train(name2idx, idx2name):
    file2idx, file_name = file2index(config.train_label, name2idx)
    kf = KFold(5, True, random_state=2019)
    fold = 0
    for train_index, test_index in kf.split(file_name):
        train = []
        val = []
        for i in train_index:
            train.append(file_name[i])
        for i in test_index:
            val.append(file_name[i])
        wc=count_labels(val,file2idx)
        print(wc)
        dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
        torch.save(dd, config.train_data + str(fold) + '.pth')
        fold = fold + 1


if __name__ == '__main__':
    pass
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)
