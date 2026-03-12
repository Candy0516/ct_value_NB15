import os
import csv
import pandas as pd
import numpy as np
from collections import Counter

def listtocsv(nanValuesList, nanIndicesList, path, col):
    if not nanValuesList:
        return
    os.makedirs(path, exist_ok=True)
    nanValuesCounter = Counter(nanValuesList)
    nanValuesCounter = sorted(nanValuesCounter.items(), key=lambda x: x[1], reverse=True)
    file_path = os.path.join(path, col + '.csv')
    
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ['ori', 'count', 'indices'])
        writer.writeheader()
        for v, c in nanValuesCounter:
            indices = [str(i) for i, val in zip(nanIndicesList, nanValuesList) if val == v]
            writer.writerow({'ori': v, 'count': c, 'indices': ';'.join(indices)})

def find_nearest_key(series_index, target_value):
    """
    從 map table 的 index 中找到最接近 target_value 的 key。
    """
    numeric_keys = pd.to_numeric(series_index, errors='coerce')
    numeric_keys = numeric_keys.dropna()
    if len(numeric_keys) == 0:
        return target_value
    numeric_array = numeric_keys.to_numpy()
    return numeric_array[np.abs(numeric_array - target_value).argmin()]

def map_testset(file_path, save_path, label_column, map_path, print=print):
    test = pd.read_csv(file_path, low_memory=False)
    column_list = test.columns.to_list()
    #map_path = os.path.join('data', '2_score', 'statistic', 'map_table')

    pvalue_test = test.copy()
    benign_test = test.copy()
    total_missing = {}

    for c, col in enumerate(column_list):
        if col in label_column:
            continue
        print(f'replace {c} {col}')
        pvalueDataFilenameByColumn = os.path.join(map_path, str(c) + '_' + col + '.npy')
        pvalueNParray = np.load(pvalueDataFilenameByColumn)
        pvalueSeries = pd.Series(pvalueNParray[1, :])
        pvalueSeries.index = pvalueNParray[0, :]

        # 原始 map 結果
        mapped_series = test[col].map(pvalueSeries)
        nan_mask = mapped_series.isna()

        # 統計缺值與索引
        nanValuesList = test.loc[nan_mask, col].tolist()
        nanIndicesList = test.loc[nan_mask].index.tolist()
        total_missing[col] = len(nanValuesList)
        listtocsv(nanValuesList, nanIndicesList, os.path.join(save_path, 'missing'), str(c) + '_' + col)

        # 對每個 NaN 找最接近的 map key，再轉為對應值
        for idx in nanIndicesList:
            original_val = test.at[idx, col]
            nearest_key = find_nearest_key(pvalueSeries.index, original_val)
            try:
                mapped_value = float(pvalueSeries[str(nearest_key)])
            except KeyError:
                mapped_value = np.nan  # 若 key 轉字串失敗（極少數情況）
            pvalue_test.at[idx, col] = mapped_value

        # 其他非 NaN 值可直接 map
        pvalue_test.loc[~nan_mask, col] = mapped_series[~nan_mask].astype(float)
        benign_test[col] = pvalue_test[col] - 0.5

    pvalue_test.to_csv(os.path.join(save_path, 'pvalue_test.csv'), index=None)
    benign_test.to_csv(os.path.join(save_path, 'benign_test.csv'), index=None)

    df_len = len(test)
    with open(os.path.join(save_path, 'missing', 'total.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, ['column', 'count', 'proportion'])
        writer.writeheader()
        for v, c in total_missing.items():
            writer.writerow({'column': v, 'count': c, 'proportion': (c / df_len) * 100})