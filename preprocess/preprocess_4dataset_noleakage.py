import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE  # 上采样，提升样本数较少的资料
from imblearn.under_sampling import TomekLinks, RandomUnderSampler  # 下采样，删除一些边界辨识度不高的样本

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def read_csv():
    """
    初始化资料集
    """

    # 读取资料集
    dataframe1 = pd.read_csv("UNSW-NB15_1.csv",
                             header=None, low_memory=False)
    dataframe2 = pd.read_csv("UNSW-NB15_2.csv",
                             header=None, low_memory=False)
    dataframe3 = pd.read_csv("UNSW-NB15_3.csv",
                             header=None, low_memory=False)
    dataframe4 = pd.read_csv("UNSW-NB15_4.csv",
                             header=None, low_memory=False)

    # 合并成一个资料集
    dataframe = pd.concat([dataframe1, dataframe2, dataframe3, dataframe4], ignore_index=True)

    return dataframe


def extract_features():
    """
    取出资料特征
    """

    feature_info = pd.read_csv(
        "NUSW-NB15_features.csv", encoding="ISO-8859-1", header=None).values
    features = feature_info[1:, 1]  # 特征名称
    feature_types = np.array([item.lower()
                             for item in feature_info[1:, 2]])  # 特征的数据类型

    # 按特征的数据类型分组 (输出为索引)
    nominal_cols = np.where(feature_types == "nominal")[0]  # 名词
    integer_cols = np.where(feature_types == "integer")[0]  # 整数
    binary_cols = np.where(feature_types == "binary")[0]  # 二进制
    float_cols = np.where(feature_types == "float")[0]  # 浮点数

    # 将不同数据类型的特征分成组
    nominal_feature = features[nominal_cols]
    integer_feature = features[integer_cols]
    binary_feature = features[binary_cols]
    float_feature = features[float_cols]

    return nominal_cols, integer_cols, binary_cols, float_cols


def process_dataset():
    """
    数据处理
    """

    dataframe = read_csv()
    nominal_cols, integer_cols, binary_cols, float_cols = extract_features()

    # 每一行都转换为同一种数值类型，无效解析设置为NaN
    dataframe[integer_cols] = dataframe[integer_cols].apply(
        pd.to_numeric, errors='coerce').astype(np.float32)
    dataframe[binary_cols] = dataframe[binary_cols].apply(
        pd.to_numeric, errors='coerce').astype(np.float32)
    dataframe[float_cols] = dataframe[float_cols].apply(
        pd.to_numeric, errors='coerce').astype(np.float32)

    # 去除空值
    # 第48列(攻击类别的名称): 1、把NaN替换为"normal"; 2、把"backdoors"替换成"backdoor"
    dataframe.loc[:, 47] = dataframe.loc[:, 47].replace(np.nan, 'normal', regex=True).apply(
        lambda x: x.strip().lower())
    dataframe.loc[:, 47] = dataframe.loc[:, 47].replace('backdoors', 'backdoor', regex=True).apply(
        lambda x: x.strip().lower())
    # 数字列: 把NaN替换为0
    dataframe.loc[:, integer_cols] = dataframe.loc[:,
                                                   integer_cols].replace(np.nan, 0, regex=True)
    dataframe.loc[:, binary_cols] = dataframe.loc[:,
                                                  binary_cols].replace(np.nan, 0, regex=True)
    dataframe.loc[:, float_cols] = dataframe.loc[:,
                                                 float_cols].replace(np.nan, 0, regex=True)
    # 名词列: 删除字符串前后空白，并将其改成小写
    dataframe.loc[:, nominal_cols] = dataframe.loc[:,
                                                   nominal_cols].applymap(lambda x: x.strip().lower())

    # 修改列名
    dataframe.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
                         'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
                         'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime',
                         'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
                         'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
                         'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

    # 删除"源IP地址"和"目标IP地址"两列 (官方给的训练集中没有这两列)
    dataframe = dataframe.drop(['srcip', 'dstip'], axis=1)

    return dataframe


def label_encoding():
    dataframe = process_dataset()

    for column in ['proto', 'service', 'state']:
        le = preprocessing.LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

def fit_minmax(X_train):
    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    return X_train_scaled, scaler

def transform_minmax(X, scaler):
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    return X_scaled

def under_sample(dataframe):
    X = dataframe.drop(['attack_cat', 'label'], axis=1)
    y = dataframe['label']
    
    X, y = RandomUnderSampler().fit_resample(X, y)

    return X, y


def label_average_process(dataframe):
    """label平均处理"""

    X = dataframe.drop(['attack_cat', 'label'], axis=1)  # 删除两列
    X = X.astype(np.float32)
    y = dataframe['label']

    # 训练集label做平均处理
    print("未处理的y_train各元素个数: ")
    print(y.value_counts(), '\n')

    print("上采样后y_train各元素个数: ")
    X, y = SMOTE().fit_resample(X, y)  # 上采样，提升样本数较少的资料
    print(y.value_counts(), '\n')

    return X, y


#訓練測試

# =====================================================
#  Split: Train / Val / Test = 60% / 20% / 20%
# =====================================================
data = label_encoding()
X = data.drop(['attack_cat', 'label'], axis=1)
y = data['label']
# Step 1: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Step 2: 20% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

X_train, scaler = fit_minmax(X_train)
X_val = transform_minmax(X_val, scaler)
X_test = transform_minmax(X_test, scaler)

# 把 label 加回去
train = X_train.copy()
train['label'] = y_train.values

val = X_val.copy()
val['label'] = y_val.values

test = X_test.copy()
test['label'] = y_test.values

print("Split result:")
print("Train:", Counter(train['label']))
print("Val  :", Counter(val['label']))
print("Test :", Counter(test['label']))

# =====================================================
#  Save CSV
# =====================================================
train.to_csv("processed/train_4noleak.csv", index=False)
val.to_csv("processed/val_4noleak.csv", index=False)
test.to_csv("processed/test_4noleak.csv", index=False)
