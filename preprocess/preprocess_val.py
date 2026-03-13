import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import KMeansSMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from KmeansSmote_binary import KMeansSMOTE_binary

seed = 7709
np.random.seed(seed)

def minmax_process(X):
    minmax_scaler = preprocessing.MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    X_minmax = pd.DataFrame(X_minmax, columns=X.columns, index=X.index)
    return X_minmax

#原程式
'''
def Sample(dataframe, sample_type):
    X = dataframe.drop(['label'], axis=1)
    y = dataframe['label']
    if sample_type == 'under':
        sampler = RandomUnderSampler(random_state=seed)
    elif sample_type == 'over':
        sampler = RandomOverSampler(random_state=seed)
    elif sample_type == 'smote':
        sampler = SMOTE(random_state=seed)
    elif sample_type == 'none':
        return X, y
    X, y = sampler.fit_resample(X, y)
    return X, y
'''

#加入KMeansSMOTE
def Sample(dataframe, sample_type):
    X = dataframe.drop(['label'], axis=1)
    y = dataframe['label']

    if sample_type == 'under':
        sampler = RandomUnderSampler(random_state=seed)
        X_res, y_res = sampler.fit_resample(X, y)

    elif sample_type == 'over':
        sampler = RandomOverSampler(random_state=seed)
        X_res, y_res = sampler.fit_resample(X, y)

    elif sample_type == 'smote':
        sampler = SMOTE(random_state=seed)
        X_res, y_res = sampler.fit_resample(X, y)

    elif sample_type == 'KMeansSMOTE_binary':
        # ✅ 呼叫你寫好的流程，傳入 dataframe（視為 train_df）
        resampled_train, _ = KMeansSMOTE_binary(dataframe.copy(), dataframe.copy(), label_column='label', exclude_type='0')
        X_res = resampled_train.drop(columns=['label'])
        y_res = resampled_train['label']

    elif sample_type == 'none':
        return X, y

    return X_res, y_res

def label_encoding(dataframe, columns):
    for column in columns:
        le = preprocessing.LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

def preprocess(filepath, filename):
    dataframe = pd.read_csv(os.path.join(filepath, filename), encoding="ISO-8859-1", low_memory=False)
    dataframe.fillna(value=0, inplace=True)

    print("----------preprocess-----------")
    all_labels = ['Dead within 24hr', 'Dead within 72hr', 'Dead within 168hr', 'Finally dead']
    
    for dead_label in range(len(all_labels)):
        print('\n' + '='*50)
        print(f'\033[35mRunning for label index {dead_label}: {all_labels[dead_label]}\033[0m')

        labels = all_labels.copy()
        label = labels.pop(dead_label)

        print('\033[36mlabel = {}\033[0m'.format(label))
        drop_columns = ['IDCODE','OPDNO','EMGADMDAT','EMGDGDAT','CSN','ITID','DGSTSID',
                        'EMGDEAD','HSPDEAD','DEADDAT','DEADSINCEEMG','HSPADMDAT','HSPDGDAT']
        drop_columns.extend(labels)
        temp_df = dataframe.copy()
        temp_df.drop(labels=drop_columns, inplace=True, axis=1)

        label_encoding(temp_df, ['RGSDPT'])
        temp_df.rename(columns={label: 'label'}, inplace=True)
        temp_df['label'] = temp_df['label'].astype(int)

        columns = temp_df.columns.tolist()
        for i, col in enumerate(columns):
            columns[i] = col.replace('/', '.') if '/' in col else col

        temp_df.columns = columns
        columns.remove('label')
        columns.append('label')
        temp_df = temp_df[columns]
        print('\033[32mdone\033[0m')

        print('\033[33mTrain / Validation / Test Split:\033[0m')
        full_train = temp_df[temp_df['Train.Test'] == 0].drop(labels=['Train.Test'], axis=1)
        test = temp_df[temp_df['Train.Test'] == 1].drop(labels=['Train.Test'], axis=1)

        # 切出 20% validation
        train, val = train_test_split(full_train, test_size=0.2, random_state=seed, stratify=full_train['label'])
        print('\033[32mdone\033[0m')

        # Test set 前處理
        print('\033[33mPreprocess Test Set:\033[0m')
        test_label = test['label']
        test.drop(labels=['label'], inplace=True, axis=1)
        test = minmax_process(test)
        test['label'] = test_label
        print('\033[32mdone\033[0m')

        # Validation set 前處理
        print('\033[33mPreprocess Validation Set:\033[0m')
        val_label = val['label']
        val.drop(labels=['label'], inplace=True, axis=1)
        val = minmax_process(val)
        val['label'] = val_label
        print('\033[32mdone\033[0m')

        sample_types = ['none', 'under', 'over', 'smote', 'KMeansSMOTE_binary']
        for sample_type in sample_types:
            print(f'\033[36m    Type: {sample_type}')
            sampled_df, sampled_label = Sample(dataframe=train.copy(), sample_type=sample_type)
            sampled_df = minmax_process(sampled_df)
            sampled_df['label'] = sampled_label
            print('    Train Label Count:')
            print(f'        {Counter(sampled_label)}')
            print('    Val Label Count:')
            print(f'        {Counter(val_label)}')
            print('    Test Label Count:')
            print(f'        {Counter(test_label)}\n\033[0m')

            # 儲存結果到對應目錄
            save_dir = os.path.join(filepath, f'preprocess_add_val_label_{dead_label}', sample_type)
            os.makedirs(save_dir, exist_ok=True)
            sampled_df.to_csv(os.path.join(save_dir, 'train.csv'), index=None)
            val.to_csv(os.path.join(save_dir, 'val.csv'), index=None)
            test.to_csv(os.path.join(save_dir, 'test.csv'), index=None)

            filepath1 = "D:/candy/北醫/new/data/1_preprocess"
            save_dir1 = os.path.join(filepath1, all_labels[dead_label], sample_type)
            os.makedirs(save_dir1, exist_ok=True)
            sampled_df.to_csv(os.path.join(save_dir1, 'train.csv'), index=None)
            val.to_csv(os.path.join(save_dir1, 'val.csv'), index=None)
            test.to_csv(os.path.join(save_dir1, 'test.csv'), index=None)

            print(f'\033[32mFinished label {dead_label}\033[0m')

if __name__ == '__main__':
    filepath = "D:/candy/北醫/new/data/0_ori_maingroup"
    filename = 'maingroup.csv'
    preprocess(filepath, filename)
