import pandas as pd
import os
import numpy as np
from sklearn import metrics
import pickle
import tqdm
import lightgbm as lgb
import warnings
import catboost as cat
warnings.simplefilter(action='ignore')
warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_acc2(pred, data_vali):
    real = data_vali.get_label()
    p = np.round(pred)
    # fpr, tpr, thresholds = metrics.roc_curve(real, pred, pos_label=1)
    return 'auc', metrics.accuracy_score(real, p), True


def self_metric(labels, preds):
    preds = preds.get_label()
    score = 1-np.mean(np.abs(labels-preds)/(np.abs(labels)+np.abs(preds)+0.1))
    return 'self_metric', score, True


def fscore(y_true, y_pred, threshold):
    y_pred2 = y_pred.copy()
    for i in range(34):
        y_pred2[:,i] = (y_pred2[:,i] > threshold[i])
    # print(y_pred)
    pred_true_sample = np.sum(np.round(np.clip(y_true * y_pred2, 0, 1)), axis=0)
    pred_sampe = np.sum(np.round(y_pred2), axis=0)
    true_sample = np.sum(np.round(y_true), axis=0)
    p = np.sum(pred_true_sample) / (np.sum(pred_sampe) + 0.00001)
    r = np.sum(pred_true_sample) / (np.sum(true_sample) + 0.00001)
    f = 2 * p * r / (p + r + 0.00001)
    return np.mean(f), pred_true_sample, pred_sampe, true_sample


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


use_age_sex = 1
is_test = 0
label_file = 'hf_round2_train.txt'
arrythmia = 'hf_round2_arrythmia.txt'
base_path1 = 'DL_feature1'
base_path3 = 'prob'
base_path2 = 'idx'
label_num = 34
get_prob = 0
add_ml_feature = 1

col = ['File_name', 'age', 'sex', 0, 1, 2, 3, 4, 5]
label = pd.read_csv(label_file, delimiter='\t', header=None, names=col) # delimiter='\t'
# label.columns = ['File_name', 'age', 'sex', 0, 1, 2, 3, 4, 5, 6, 7, 8]
# label.drop([8], axis=1, inplace=True)
label_map = pd.read_csv(arrythmia, delimiter='\t', header=None, encoding='utf-8', engine='python')
# label_map.drop([1, 2, 3], axis=1, inplace=True)
label_dict = dict(zip(label_map[0].unique(),range(label_map[0].nunique())))
label_map['code'] = label_map[0].map(label_dict)
sex_map = {'FEMALE': 0, 'MALE':1}
label['sex'] = label['sex'].map(sex_map)
for i in range(6):
    label[str(i)+'_code'] = label[i].map(label_dict)
label.drop([0, 1, 2, 3, 4, 5], axis=1, inplace=True)
filenameslist, filelabelslist, file_age, file_sex = read_list(label, label_num)
filelabelslist = np.array(filelabelslist)
label = label.rename(columns={'File_name': 'filename'})

result = []
ff = []

oof = np.zeros((len(filelabelslist), label_num))

if add_ml_feature == 1:
    data2 = pd.read_csv('train_ml.csv', header=0, engine='python')

model_name = 'cat'
mask_value = -9999
feature_len = 512
feature_len2 = 1024
prob_dl = np.zeros((len(label), label_num))
#dl_name2 = 'resnet34'
dl_name1 = 'myecgnet'
# my_file = pd.read_csv('mylabel.csv')
# my_file = list(my_file['File_name'])

for i in range(5):
#    if i not in [0]:
#        continue
#     read data
    print('-----------' + str(i) + '---------------')
    # data = pd.read_csv(os.path.join(base_path1, str(i) + '_train.csv'), header=0, engine='python')
    # data = pd.read_csv(os.path.join(base_path3, 'train_prob' + str(i) + '.csv'), header=0, engine='python')
    ### --------------------1
    ###
    data = pd.read_csv(os.path.join(base_path1, 'prob_'+dl_name1 + '_' + 
                                    str(i)+'.csv'), header=0, engine='python')
    col1 = [str(i) for i in range(34)] + ['filename']
    data = data[col1]
    col1 = ['prob1_'+str(i) for i in range(34)] + ['filename']
    data.columns = col1
    col1 = ['prob1_'+str(i) for i in range(34)]
    # if i not in [0, 2, 6, 7, 8, 14, 15, 16, 19, 21, 23, 25, 32, 33]
    
    data_2 = pd.read_csv(os.path.join(base_path1, dl_name1 + '_' + 
                                      str(i) + '_test.csv'), header=0, engine='python')
    col2 = [str(i) for i in range(feature_len)] + ['filename']
    data_2 = data_2[col2]
    col2 = ['df1_'+str(i) for i in range(feature_len)] + ['filename']
    data_2.columns = col2
    col2 = ['df1_'+str(i) for i in range(feature_len)]
#    
    data = data.merge(data_2, 'left', on='filename')
    
    ###------------------2
    ###
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

    
    if use_age_sex == 1:
        data = data.merge(label[['filename', 'age', 'sex']], 'left', on='filename')
    if add_ml_feature == 1:
        data = data.merge(data2[[str(col) for col in range(1336)] + ['filename']],
                          'left', on='filename')
    names = data['filename']
#    if add_ml_feature == 1:
#        data = data.merge(data2[[str(col) for col in range(1336)] + ['filename']],
#                          'left', on='filename')

    # col = [tmp_col for tmp_col in data.columns if tmp_col not in ['filename']]
#    if add_ml_feature == 1:
#        col = [str(ii)+'_x' for ii in range(256)] + ['age', 'sex'] + [str(ii)+'_y' for ii in range(256)] + \
#            [str(ii) for ii in range(256, 1336)]
#    else:
#        # col = [tmp_col for tmp_col in data.columns if tmp_col not in ['filename', 'index']]
#        col = [str(tmp_col) for tmp_col in range(34)] + ['age', 'sex']
#    col = [str(ii)+'_x' for ii in range(256)] + ['age', 'sex'] + [str(ii)+'_y' for ii in range(256)] + \
#        [str(ii) for ii in range(256, 1336)]
    col = col1 + col2 + [str(i) for i in range(1336)] + ['age', 'sex']
#    col = col1 + ['age', 'sex']
    trn_idx = np.load(os.path.join(base_path2,  'train'+str(i)+'.npy'))
    val_idx = np.load(os.path.join(base_path2, 'val'+str(i)+'.npy'))
    
    trn_idx = data['filename'].isin(trn_idx)
    val_idx = data['filename'].isin(val_idx)
    
#    prob_dl[val_idx] = sigmoid(data[['prob1_'+str(i) for i in range(34)]].values[val_idx])
    
    data = data[col].values
#    print(data.shape)
    # read idx

    # val_idx = [ii for ii in range(i, len(filenameslist), 5)]
    # trn_idx = [ii for ii in range(len(filenameslist)) if ii not in val_idx]

    # for ii in trn_idx.copy():
    #     if filenameslist[ii] not in my_file:
    #         trn_idx.remove(ii)
    # for ii in val_idx.copy():
    #     if filenameslist[ii] not in my_file:
    #         val_idx.remove(ii)

    trn_data = data[trn_idx]
    val_data = data[val_idx]
#    print(val_data.shape, trn_data.shape)
    trn_label = filelabelslist[trn_idx]
    val_label = filelabelslist[val_idx]
    pred = np.zeros(val_label.shape)
    tmp_result = []
    
    for j in range(label_num):
#        if j not in [17, 18, 20, 27, 28, 30, 31]:
#            continue
#        print('folds: %d, label: %d' % (i, j))
        if use_age_sex == 0:
            save_base_path = os.path.join('model', str(i))
        else:
            save_base_path = os.path.join('model', str(i))
        if os.path.exists(save_base_path) == False:
            os.mkdir(save_base_path)

        params = {
            'learning_rate': 0.05,
            'boosting_type': 'dart',
            'objective': 'binary',
#            'metric': 'binary_logloss',
#            'metric': 'auc',
            'num_leaves': 31,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            # 'is_unbalance': True,
            'seed': 1,
            'bagging_seed': 1,
            'feature_fraction_seed': 7,
            'min_data_in_leaf': 5,
            'nthread': -1
        }
#       
#        if j in [17, 18, 20, 30]:
##            print(trn_data[:,len(col1):].shape)
#            dtrain = lgb.Dataset(trn_data[:,len(col1):], label=trn_label[:, j])
#            dvalid = lgb.Dataset(val_data[:,len(col1):], label=val_label[:, j])
#        elif j in [27, 28, 31]:
#            col_idx = [i for i in range(len(col1))] + [i+len(col1)+len(col2) for i in range(1338)]
##            print(trn_data[:, col_idx].shape)
#            dtrain = lgb.Dataset(trn_data[:, col_idx], label=trn_label[:, j])
#            dvalid = lgb.Dataset(val_data[:, col_idx], label=val_label[:, j])
#        else:
        dtrain = lgb.Dataset(trn_data, label=trn_label[:, j])
        dvalid = lgb.Dataset(val_data, label=val_label[:, j])
        save_path = os.path.join(save_base_path, 'model_' + model_name + str(j) + '.pickle')
        if is_test == 0:
            if model_name == 'lgb':
                clf = lgb.train(
                    params=params,
                    train_set=dtrain,
                    num_boost_round=10000,
                    valid_sets=[dvalid],
                    early_stopping_rounds=100,
                    verbose_eval=300,
                    feval = self_metric
                )
            if model_name == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, max_depth=11)
                trn_data[np.isnan(trn_data)] = mask_value
                clf.fit(trn_data, trn_label[:, j])
            if model_name == 'cat':
                clf = cat.CatBoostClassifier(task_type='GPU', devices='0', 
                                             eval_metric='F1', use_best_model=True, 
                                             early_stopping_rounds=500, random_state=2019, 
                                             boosting_type='Plain', logging_level='Silent')
#                trn_data[np.isnan(trn_data)] = mask_value
                clf.fit(trn_data, trn_label[:, j], eval_set=(val_data, val_label[:, j]), verbose=None)
        else:
#            clf = cat.CatBoostClassifier(task_type='GPU', devices='0')
#            clf.load_model(save_path)
            with open(save_path, 'rb+') as f:
                clf = pickle.load(f)
#        break
        if model_name == 'lgb':
#            if j in [17, 18, 20, 30]:
#                oof[val_idx, j] = clf.predict(val_data[:,len(col1):], num_iteration=clf.best_iteration)
#            elif j in [27, 28, 31]:
#                oof[val_idx, j] = clf.predict(val_data[:, col_idx], num_iteration=clf.best_iteration)
#            else:
            oof[val_idx, j] = clf.predict(val_data, num_iteration=clf.best_iteration)
        elif model_name == 'rf':
            val_data[np.isnan(val_data)] = mask_value
            oof[val_idx, j] = clf.predict(val_data)
        elif model_name == 'cat':
#            val_data[np.isnan(val_data)] = mask_value
            oof[val_idx, j] = clf.predict(val_data)
            
        pred[:, j] = np.round(oof[val_idx, j])
        # print(metrics.accuracy_score(val_label[:, j], pred[:, j]))
#        oof[trn_idx, j] = clf.predict(trn_data, num_iteration=clf.best_iteration)
#        tmp_result.append(metrics.accuracy_score(val_label[:, j], pred[:, j]))

        if is_test == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(clf, f)
        
    threshold = np.ones(34) * 0.5
    # threshold[2] = 0.01
    f1, pred_true_sample, pred_sampe, true_sample = fscore(val_label, oof[val_idx], threshold)
    print(f1)
    ff.append(f1)
    result.append(tmp_result)
    # break
print(np.mean(ff))

#if get_prob:
#    df = pd.DataFrame(oof)
#    df['filename'] = names
#    df.to_csv('ML_feature/' + 'prob.csv', index=None) 

#for i in [1, 10, 12, 13]:
#    oof[:, i] = prob_dl[:, i]

#f1, pred_true_sample, pred_sampe, true_sample = fscore(filelabelslist, oof, threshold)
#df = pd.DataFrame(true_sample)
#df = pd.concat((df, pd.DataFrame(pred_true_sample)), axis=1)
#df = pd.concat((df, pd.DataFrame(pred_sampe)), axis=1)
#df.columns = ['P', 'TP', 'PP']
#df['FP'] = df['PP'] - df['TP']
#df['RN'] = len(filelabelslist) -df['P']
#df['TN'] = df['RN'] - df['FP']
#df['FN'] = df['P'] - df['TP']
#df.to_csv('result'+'.csv')
#print(f1)
