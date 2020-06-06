import pandas as pd
import os
import numpy as np
import pickle
import tqdm
from config_lgb import config

fold = [0]

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
# sub.columns = ['filename', 'age', 'sex']
sub['sex_new'] = sub['sex'].map(sex_map)
# df = pd.DataFrame()
result = []
add_ml_feature = 0

if add_ml_feature == 1:
    data2 = pd.read_csv(os.path.join('hefei_ecg', 'test_biosppy.csv'), header=0, engine='python')
pred = np.zeros((len(sub), label_num))

for i in fold:
    # read data
    print('-----------' + str(i) + '---------------')
    df = sub.copy()
    data = pd.read_csv(os.path.join(base_path1, str(i) + config.feature), header=0, engine='python')
    # data = pd.read_csv(os.path.join(base_path1, 'test_prob.csv'), header=0, engine='python')

    if add_ml_feature == 1:
        data = data.merge(data2[[str(col) for col in range(1336)] + ['filename']],
                          'left', on='filename')

    df = df.merge(data, 'left', on='filename')

    if add_ml_feature == 1:
        col = [str(ii) + '_x' for ii in range(256)] + ['age', 'sex_new'] + [str(ii) + '_y' for ii in range(256)] + \
              [str(ii) for ii in range(256, 1336)]

    else:
        col = [str(tmp) for tmp in range(1024)]
        if use_age_sex:
            col = col + ['age', 'sex_new']

    # df['idx'] = df['filename'].apply(lambda x: x[:-4])
    # col = [tmp_col for tmp_col in data.columns if tmp_col not in ['filename']]
    # col = col + ['idx']

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

        pred_fold[:, j] = clf.predict(val_data, num_iteration=clf.best_iteration)
    pred = pred + pred_fold
pred = pred / len(fold)

# df = pd.DataFrame(pred, columns=[i for i in range(label_num)])
# df['filename'] = sub['filename']
# df.to_csv(os.path.join('prob', 'test.csv'), index=None)

pred = np.round(pred)

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
