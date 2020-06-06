# -*- coding：utf-8 -*-
# Author: juzstu
# Time: 2019/10/19 11:27

import pandas as pd


def get_file_from_round1(file_path, round2_label, kind, myfile):
    with open(file_path, encoding='utf8') as f:
        file_list = []
        for i in f.readlines():
            tmp = [j for j in i.split('\t')[3:] if j in round2_label and j != '\n']
            if len(tmp) > 0 and i.split('\t')[0] in myfile:
                file_list.append('\t'.join(i.split('\t')[:3]+tmp)+'\t' + str(kind))
                # file_list.append('\t' + str(kind))
        return file_list


if __name__ == "__main__":
    label = pd.read_csv('hf_round2_arrythmia.txt', header=None)[0].values
    mylabel = pd.read_csv('mylabel.csv')
    myfile = mylabel['File_name'].values
    train_round1 = get_file_from_round1('hf_round1_label.txt', label, 1, myfile)
    round1_subA = get_file_from_round1('heifei_round1_ansA_20191008.txt', label, 2, myfile)
    train_round1.extend(round1_subA)
    with open('round_1_data_for_round2.txt', 'w', encoding='utf8') as r1:
        for r in train_round1:
            r1.write(r + '\n')
