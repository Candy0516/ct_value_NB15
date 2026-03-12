from collections import Counter
import pandas as pd
import numpy as np
import os
#ct值=p值-0.5
'''
def score(file_path, save_path, label_column, print=print):
    # ========== 載入資料 ==========
    data = pd.read_csv(file_path)
    column_list = data.columns.to_list()
    os.makedirs(os.path.join(save_path, 'statistic', 'map_table'), exist_ok=True)

    # ========== 計算黑白樣本數與比例 ==========
    n_black = int((data['label'] == 1).sum())
    n_white = int((data['label'] == 0).sum())

    if n_black >= n_white:
        black_more = True
        ratio = n_black / max(n_white, 1)
    else:
        black_more = False
        ratio = n_white / max(n_black, 1)

    # ========== 逐欄位處理 ==========
    for c, col in enumerate(column_list):
        if col in label_column:
            continue
        print(f"{c} {col}")

        # 載入指定欄位與標籤
        df = pd.read_csv(file_path, usecols=[col, 'label'], low_memory=False)
        df[col] = df[col].astype(str)

        # 計算出現次數（全體與良性）
        full_count = df[col].value_counts()
        benign_count = df[df['label'] == 0][col].value_counts()

        # 合併為 DataFrame，統一計算
        map_df = pd.DataFrame({
            "feature_value": full_count.index,
            "full_count": full_count.values,
            "benign_count": benign_count.reindex(full_count.index).fillna(0).values
        })

        # 確保 feature_value 為字串（避免型別 mismatch）
        map_df["feature_value"] = map_df["feature_value"].astype(str)

        # 計算調整後的 benign、malicious、total
        if black_more:
            map_df["malicious_count"] = map_df["full_count"] - map_df["benign_count"]
            map_df["benign_adj"] = map_df["benign_count"] * ratio
        else:
            map_df["malicious_count"] = (map_df["full_count"] - map_df["benign_count"]) * ratio
            map_df["benign_adj"] = map_df["benign_count"]

        map_df["full_adj"] = map_df["benign_adj"] + map_df["malicious_count"]

        # 避免除以 0，預設 ct=0
        map_df["ct_value"] = 0.0
        valid_mask = map_df["full_adj"] > 0
        map_df.loc[valid_mask, "ct_value"] = (
            np.log(map_df.loc[valid_mask, "full_adj"] + 1) *
            ((map_df.loc[valid_mask, "benign_adj"] / map_df.loc[valid_mask, "full_adj"]) - 0.5)
        )

        # ✅ 改成純 Python list 建立 numpy array：
        feature_value_list = map_df["feature_value"].astype(str).tolist()
        ct_value_list = map_df["ct_value"].astype(float).tolist()

        array = np.array([feature_value_list, ct_value_list])  # 確保是基本型別組成的陣列
        save_file = os.path.join(save_path, 'statistic', 'map_table', str(c)+'_'+col+'.npy')
        np.save(save_file, array)
'''

def score(file_path, save_path, label_column, print=print):
    # 讀檔
    print(file_path)
    data = pd.read_csv(file_path)
    column_list = data.columns.to_list()

    os.makedirs(os.path.join(save_path, 'statistic', 'map_table'), exist_ok=True)

    benign_count = 0
    malicious_count = 0
    for label in data['label']:
        if label == 0:
            benign_count += 1
        else:
            malicious_count += 1
    if malicious_count>=benign_count:
        black_more = True
        ratio = malicious_count/max(benign_count, 1)
    else:
        black_more = False
        ratio = benign_count/max(malicious_count, 1)

    
    # 對每個feature計算各個特徵值出現的機率
    for c, col in enumerate(column_list):
        if col in label_column: continue
        print(str(c) + ' ' + col)
        data = pd.read_csv(file_path, usecols=[col, 'label'], low_memory=False)

        # 特徵值出現的次數（全部）
        feature_counts = Counter(data[column_list[c]])

        # 良性樣本的次數
        benign = data[data['label'] == 0]
        benign_counts = Counter(benign[column_list[c]])

        # 儲存 CT 值
        balance_pvalue = {}

        for key, value in feature_counts.items():
            benign = benign_counts.get(key, 0)

            if black_more:
                malicious = value - benign
                benign_adj = benign * ratio
            else:
                malicious = (value - benign) * ratio
                benign_adj = benign

            full_adj = benign_adj + malicious

            # if full_adj == 0:
            #     ct = 0.0
            # else:
            #     ct = np.log(full_adj + 1) * ((benign_adj / full_adj) - 0.5)
            ct = np.log(full_adj + 1) * ((benign_adj / full_adj) - 0.5)

            balance_pvalue[key] = ct + 0.5


        # 使用numpy儲存
        array = np.array([list(balance_pvalue.keys()), list(balance_pvalue.values())])
        np.save(os.path.join(save_path, 'statistic', 'map_table', str(c)+'_'+col+'.npy'), array)
