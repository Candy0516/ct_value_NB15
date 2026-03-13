# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:52:20 2017

@author: Samuli
"""


import numpy as np

import pandas 

from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def label_encode_nominal(
    dataframe,
    nominal_columns,
    encoders=None,
    fit=True,
    fill_value="unknown"
):

    if encoders is None:
        encoders = {}

    encoded_cols = []

    for col in nominal_columns:
        if col not in dataframe.columns:
            continue

        # 1️⃣ 補缺失值 + 轉字串
        values = dataframe[col].fillna(fill_value).astype(str)

        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(values)
            encoders[col] = le
        else:
            le = encoders[col]
            encoded = le.transform(values)

        encoded_cols.append(encoded.reshape(-1, 1))

    # 若沒有任何 nominal 特徵
    if len(encoded_cols) == 0:
        nominal_x = np.empty((len(dataframe), 0), dtype=np.float32)
    else:
        nominal_x = np.hstack(encoded_cols).astype(np.float32)

    return nominal_x, encoders

# Read the info about features
print('Reading feature info...')
data_info = pandas.read_csv("NUSW-NB15_features.csv", encoding = "ISO-8859-1", header=None).values
features = data_info[1:-2,:]
feature_names = features[:, 1]  # Names of the features in a list
feature_types = np.array([item.lower() for item in features[:, 2]])  # The types of the corresponding features in 'features_names'
                         
# 原始 feature names
feature_names_raw = feature_names

# 正規化：小寫 + 去除前後空白
feature_names_norm = np.array([
    str(f).strip().lower()
    for f in feature_names_raw
])

#名稱修正
alias_map = {
    "dintpkt": "dinpkt",
    "sintpkt": "sinpkt",
    "smeansz": "smean",
    "dmeansz": "dmean",
    "ct_src_ ltm": "ct_src_ltm",
    "res_bdy_len": "response_body_len"
}

feature_names_norm = np.array([
    alias_map.get(f, f) for f in feature_names_norm
])

# arrays for names of the different types of features
nominal_names = feature_names_norm[feature_types == "nominal"]
integer_names = feature_names_norm[feature_types == "integer"]
float_names   = feature_names_norm[feature_types == "float"]
binary_names  = feature_names_norm[feature_types == "binary"]
print(nominal_names)


print('Reading csv files...')
'''
dataframe1 = pandas.read_csv("UNSW-NB15_1.csv", header=None)
dataframe2 = pandas.read_csv("UNSW-NB15_2.csv", header=None)
dataframe3 = pandas.read_csv("UNSW-NB15_3.csv", header=None)
dataframe4 = pandas.read_csv("UNSW-NB15_4.csv", header=None)

print('Concatenating...')
dataframe = pandas.concat([dataframe1, dataframe2, dataframe3, dataframe4])

del dataframe1
del dataframe2
del dataframe3
del dataframe4
'''
#正規化 dataframe 的 columns
dataframe = pandas.read_csv("UNSW_NB15_testing-set.csv")
dataframe.columns = (
    dataframe.columns
    .astype(str)
    .str.strip()
    .str.lower()
)
print(dataframe.columns)

# 只保留實際存在於 dataframe 中的特徵
nominal_names_exist = [c for c in nominal_names if c in dataframe.columns]
integer_names_exist = [c for c in integer_names if c in dataframe.columns]
float_names_exist   = [c for c in float_names   if c in dataframe.columns]
binary_names_exist  = [c for c in binary_names  if c in dataframe.columns]

#加上rate特徵
if 'rate' in dataframe.columns and 'rate' not in float_names_exist:
    float_names_exist.append('rate')

print('Preprocessing...')
print('Converting data...')
for col in integer_names:
    if col in dataframe.columns:
        dataframe[col] = pandas.to_numeric(dataframe[col], errors='coerce')

for col in float_names:
    if col in dataframe.columns:
        dataframe[col] = pandas.to_numeric(dataframe[col], errors='coerce')

for col in binary_names:
    if col in dataframe.columns:
        dataframe[col] = pandas.to_numeric(dataframe[col], errors='coerce')

print('Replacing NaNs...')
dataframe['attack_cat'] = (
    dataframe['attack_cat']
    .fillna('normal')
    .astype(str)
    .str.strip()
    .str.lower()
    .replace('backdoors', 'backdoor')
)
for col in nominal_names:
    if col in dataframe.columns:
        dataframe[col] = dataframe[col].astype(str).str.strip().str.lower()
for col in binary_names:
    if col in dataframe.columns:
        dataframe[col] = dataframe[col].fillna(0)


# dataset = dataframe.values

# del dataframe

# Subsets of the dataset which have data that is only of the corresponding data type (nominal, integer etc)
# Columns don't include the target classes (the two last columns of the dataset)
print('Slicing dataset...')
nominal_x = dataframe[nominal_names_exist].values
integer_x = dataframe[integer_names_exist].values.astype(np.float32)
float_x   = dataframe[float_names_exist].values.astype(np.float32)
binary_x  = dataframe[binary_names_exist].values.astype(np.float32)
# aon_x = dataset[:, 48][np.newaxis,:].astype(np.float32).transpose()  # Attack or not (binary)
ignored_features = set(feature_names_norm) - set(dataframe.columns)

print("\n[Ignored features not found in CSV]")
for f in sorted(ignored_features):
    print(f)

ignored_features_f =  set(dataframe.columns) - set(feature_names_norm)

print("\n[Ignored features not found in feature CSV]")
for f in sorted(ignored_features_f):
    print(f)

# Make nominal (textual) data binary vectors
print('Vectorizing nominal data...')
#train
nominal_x, label_encoders = label_encode_nominal(
    dataframe,
    nominal_names_exist,
    fit=True
)

#寫回 dataframe
for i, col in enumerate(nominal_names_exist):
    dataframe[col] = nominal_x[:, i].astype(np.int32)

#再轉成 float
for col in integer_names_exist + nominal_names_exist + float_names_exist + binary_names_exist:
    dataframe[col] = dataframe[col].astype(np.float32)
#val, test
# X_nom_val, _ = label_encode_nominal(
#     val_df,
#     nominal_names_exist,
#     encoders=label_encoders,
#     fit=False
# )
# X_nom_test, _ = label_encode_nominal(
#     test_df,
#     nominal_names_exist,
#     encoders=label_encoders,
#     fit=False
# )

print('Concatenating X...')
X = np.concatenate((integer_x, nominal_x, float_x, binary_x), axis=1)
# X_val = np.concatenate(
#     (X_int_val, X_nom_val, X_float_val, X_bin_val),
#     axis=1
# )

# X_test = np.concatenate(
#     (X_int_test, X_nom_test, X_float_test, X_bin_test),
#     axis=1
# )

del integer_x
del nominal_x
del float_x
del binary_x

# Find rows that have NaNs

print('Removing NaN rows...')
nan_rows = np.unique(np.where(np.isnan(X))[0])

X_no_nans = np.delete(X, nan_rows, axis=0)

Y_cat = dataframe['attack_cat'].values
Y_bin = dataframe['label'].values.astype(np.int16)

print('Normalizing X...')
scaler = MinMaxScaler()
normalized_X = scaler.fit_transform(X_no_nans)

del X_no_nans


# =====================================================
#  Build final feature names (order MUST match X)
# =====================================================
print('Building feature names...')

X_feature_order = (
    list(integer_names_exist) +
    list(nominal_names_exist) +
    list(float_names_exist) +
    list(binary_names_exist)
)

# =====================================================
#  Save to CSV (last column = binary label)
# =====================================================
print('Saving CSV...')
df_norm = pandas.DataFrame(
    normalized_X,
    columns=X_feature_order
)
for col in X_feature_order:
    dataframe.loc[dataframe.index.difference(nan_rows), col] = df_norm[col].values
feature_cols_in_order = [
    col for col in dataframe.columns
    if col in X_feature_order
]

df_out = dataframe.loc[:, feature_cols_in_order].copy()
df_out['label'] = dataframe['label'].values.astype(np.int16)

df_out.to_csv(
    'UNSW_NB15_binary_named_normalized.csv',
    index=False,
    encoding='utf-8'
)


# df = pandas.DataFrame(X, columns=final_feature_names)
# df['label'] = Y_bin

# df.to_csv(
#     'UNSW_NB15_binary_named.csv',
#     index=False,
#     encoding='utf-8'
# )

print('Done!')
print('Final shape:', df_out.shape)
print('Label distribution:')
print(df_out['label'].value_counts())
