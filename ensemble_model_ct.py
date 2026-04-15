import pandas as pd
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 初始化模型結果儲存
ct_results = []
xg_results = []
rf_results = []
ct_xg_results = []
ct_rf_results = []
ct_xg_rf_results = []
ct_xg_rf_ada_results = []

# 建立 result 輸出資料夾
os.makedirs('ensemble_result_dash_ct0', exist_ok=True)

#LABELS = ['Dead within 24hr', 'Dead within 72hr', 'Dead within 168hr', 'Finally dead']

def Get_Scale_Pos_Weight(data, sensitivity):
    Nn = len(data[data['label'] == 0])
    Np = len(data[data['label'] == 1])
    SPW = sensitivity * (Nn / Np)
    return SPW

"""使用Youden's J統計量尋找最佳閾值"""
def find_best_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

"""計算評估指標"""
def evaluate(y_true, y_pred, y_score, model_usage=None):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp) if (tn + fp) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    result = {
        'Accuracy': accuracy,
        'AUROC': auroc, 'AUPRC': auprc,
        'TNR': tnr, 'TPR': tpr,
        'PPV': ppv, 'NPV': npv
    }
    if model_usage:
        result.update(model_usage)
    return result

"""統計各模型被選擇的次數"""
def count_model_selections(y_true, *model_preds):
    model_names = ['ct', 'xg', 'rf', 'ada'][:len(model_preds)]
    abs_preds = [np.abs(p) for p in model_preds]
    counts = {name: 0 for name in model_names}
    for i in range(len(y_true)):
        idx = np.argmax([pred[i][0] for pred in abs_preds])
        counts[model_names[idx]] += 1
    return counts
    # 這是一個 list，裡面有 3 個 shape=(5,1) 的 NumPy array
    # abs_preds = [
    # np.array([[0.8], [0.3], [0.1], [0.7], [0.2]]),  
    # np.array([[0.6], [0.2], [0.5], [0.1], [0.4]]),  
    # np.array([[0.4], [0.6], [0.3], [0.9], [0.1]])   
    # ]
    # 第一次for pred迴圈執行:np.array([[0.8], [0.3], [0.1], [0.7], [0.2]]),
    # pred[i][0] → 正確取得 scalar 值（可比大小）
    # pred[i] → 拿到的是 list

# for label in LABELS:
#     print(f'label: {label}\n')

# 載入資料
train = pd.read_csv(f"D:/candy/TONIOT/data/1_preprocess/train.csv")
val = pd.read_csv(f"D:/candy/TONIOT/data/1_preprocess/val.csv")
test = pd.read_csv(f"D:/candy/TONIOT/data/1_preprocess/test.csv")

X_train = train.drop(columns=['label'])
y_train = train['label']
X_val = val.drop(columns=['label'])
y_val = val['label']
X_test = test.drop(columns=['label'])
y_test = test['label']

# 訓練三個模型
print('    Training model')
rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", max_depth=10, random_state=42)
xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, scale_pos_weight=Get_Scale_Pos_Weight(train, sensitivity=0.85), use_label_encoder=False, eval_metric='logloss', random_state=42)
ada = AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
ada.fit(X_train, y_train)

# 計算閾值
thresholds = {
    'rf': find_best_threshold(y_val, rf.predict_proba(X_val)[:, 1]),
    'xg': find_best_threshold(y_val, xgb.predict_proba(X_val)[:, 1]),
    'ada': find_best_threshold(y_val, ada.predict_proba(X_val)[:, 1])
}

# 產生 CT 分數
# ct_score = -pd.read_csv(f"D:/candy/TONIOT/data/7_sum_test/benign_test.csv")['sum'].values.reshape(-1, 1)
# ct_val_score = -pd.read_csv(f"D:/candy/TONIOT/data/7_sum_val/benign_val.csv")['sum'].values.reshape(-1, 1)
ct_score = -pd.read_csv(f"D:/candy/TONIOT/data/8_dash_ct_0/test.csv")['sum'].values.reshape(-1, 1)
ct_val_score = -pd.read_csv(f"D:/candy/TONIOT/data/8_dash_ct_0/val.csv")['sum'].values.reshape(-1, 1)

# 計算 CT 閾值
ct_thresh = find_best_threshold(y_val, ct_val_score.flatten())

# 標準化 + 閾值偏移
scaler_ct = MinMaxScaler().fit(ct_val_score)
scaler_rf = MinMaxScaler().fit(rf.predict_proba(X_val)[:, 1].reshape(-1, 1))
scaler_xg = MinMaxScaler().fit(xgb.predict_proba(X_val)[:, 1].reshape(-1, 1))
scaler_ada = MinMaxScaler().fit(ada.predict_proba(X_val)[:, 1].reshape(-1, 1))

#避免測試資料洩漏，使用驗證集建立標準化參數(scaler_rf)，並將該 MinMaxScaler 應用於測試集預測分數的縮放與閾值調整
ct_pred = scaler_ct.transform(ct_score) - scaler_ct.transform([[ct_thresh]])
rf_pred = scaler_rf.transform(rf.predict_proba(X_test)[:, 1].reshape(-1, 1)) - scaler_rf.transform([[thresholds['rf']]]) #增量程式有.reshape(-1, 1)但可以不加
xg_pred = scaler_xg.transform(xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)) - scaler_xg.transform([[thresholds['xg']]])
ada_pred = scaler_ada.transform(ada.predict_proba(X_test)[:, 1].reshape(-1, 1)) - scaler_ada.transform([[thresholds['ada']]])

#CT為主
# 1. CT 單獨
ct_bin = (ct_pred > 0).astype(int)
ct_metrics = evaluate(y_test, ct_bin, MinMaxScaler().fit_transform(ct_pred))

# 2. XG 單獨
xg_bin = (xg_pred > 0).astype(int)
xg_metrics = evaluate(y_test, xg_bin, MinMaxScaler().fit_transform(xg_pred))

# 2. RF 單獨
rf_bin = (rf_pred > 0).astype(int)
rf_metrics = evaluate(y_test, rf_bin, MinMaxScaler().fit_transform(rf_pred))

# 3. CT + XG
ct_xg_pred = np.where(np.abs(ct_pred) > np.abs(xg_pred), ct_pred, xg_pred)
ct_xg_bin = (ct_xg_pred > 0).astype(int)
ct_xg_metrics = evaluate(
    y_test,
    ct_xg_bin,
    MinMaxScaler().fit_transform(ct_xg_pred),
    count_model_selections(y_test, ct_pred, xg_pred)
)

# 3. CT + RF
ct_rf_pred = np.where(np.abs(ct_pred) > np.abs(rf_pred), ct_pred, rf_pred)
ct_rf_bin = (ct_rf_pred > 0).astype(int)
ct_rf_metrics = evaluate(
    y_test,
    ct_rf_bin,
    MinMaxScaler().fit_transform(ct_rf_pred),
    count_model_selections(y_test, ct_pred, rf_pred)
)

# 4. CT + XG + RF
ct_xg_rf_pred = np.where(
    (np.abs(ct_pred) > np.abs(xg_pred)) & (np.abs(ct_pred) > np.abs(rf_pred)),
    ct_pred,
    np.where(np.abs(xg_pred) > np.abs(rf_pred), xg_pred, rf_pred)
)
ct_xg_rf_bin = (ct_xg_rf_pred > 0).astype(int)
ct_xg_rf_metrics = evaluate(
    y_test,
    ct_xg_rf_bin,
    MinMaxScaler().fit_transform(ct_xg_rf_pred),
    count_model_selections(y_test, ct_pred, xg_pred, rf_pred)
)

# 5. CT + XG + RF + ADA
ct_xg_rf_ada_pred = np.where(
    (np.abs(ct_pred) > np.abs(xg_pred)) &
    (np.abs(ct_pred) > np.abs(rf_pred)) &
    (np.abs(ct_pred) > np.abs(ada_pred)),
    ct_pred,
    np.where(
        (np.abs(xg_pred) > np.abs(rf_pred)) & (np.abs(xg_pred) > np.abs(ada_pred)),
        xg_pred,
        np.where(np.abs(rf_pred) > np.abs(ada_pred), rf_pred, ada_pred)
    )
)
ct_xg_rf_ada_bin = (ct_xg_rf_ada_pred > 0).astype(int)
ct_xg_rf_ada_metrics = evaluate(
    y_test,
    ct_xg_rf_ada_bin,
    MinMaxScaler().fit_transform(ct_xg_rf_ada_pred),
    count_model_selections(y_test, ct_pred, xg_pred, rf_pred, ada_pred)
)

# 基本資訊
info = {
    'TestSet': 'final',
    'train_0_count': (y_train == 0).sum(),
    'train_1_count': (y_train == 1).sum(),
    'xg_thres': thresholds['xg'],
    'rf_thres': thresholds['rf'],
    'ada_thres': thresholds['ada'],
    'ct_thres': ct_thresh
}

# 儲存
ct_results.append({**info, **ct_metrics})
xg_results.append({**info, **xg_metrics})
rf_results.append({**info, **rf_metrics})
ct_xg_results.append({**info, **ct_xg_metrics})
ct_rf_results.append({**info, **ct_rf_metrics})
ct_xg_rf_results.append({**info, **ct_xg_rf_metrics})
ct_xg_rf_ada_results.append({**info, **ct_xg_rf_ada_metrics})

# 產出結果
df_ct = pd.DataFrame(ct_results)
df_ct["Model"] = "CT"

df_xg = pd.DataFrame(xg_results)
df_xg["Model"] = "XG"

df_rf = pd.DataFrame(rf_results)
df_rf["Model"] = "RF"

df_ct_xg = pd.DataFrame(ct_xg_results)
df_ct_xg["Model"] = "CT+XG"

df_ct_rf = pd.DataFrame(ct_rf_results)
df_ct_rf["Model"] = "CT+RF"

df_ct_xg_rf = pd.DataFrame(ct_xg_rf_results)
df_ct_xg_rf["Model"] = "CT+XG+RF"

df_ct_xg_rf_ada = pd.DataFrame(ct_xg_rf_ada_results)
df_ct_xg_rf_ada["Model"] = "CT+XG+RF+ADA"

# 合併儲存
all_results = pd.concat([
    df_ct,
    df_xg,
    df_rf,
    df_ct_xg,
    df_ct_rf,
    df_ct_xg_rf,
    df_ct_xg_rf_ada
], ignore_index=True)

all_results.to_csv(f"ensemble_result/ensemble_results_ctbase_split.csv", index=False)

print(f"新版融合順序結果 ensemble_results_ctbase.csv 已儲存完畢")

#模型越界圖
'''
# 標準化後的分數與閾值
ct_scaled = scaler_ct.transform(ct_score)
rf_scaled = scaler_rf.transform(rf.predict_proba(X_test)[:, 1].reshape(-1, 1))
xg_scaled = scaler_xg.transform(xgb.predict_proba(X_test)[:, 1].reshape(-1, 1))
ada_scaled = scaler_ada.transform(ada.predict_proba(X_test)[:, 1].reshape(-1, 1))
ct_thresh = scaler_ct.transform([[ct_thresh]])
rf_thresh = scaler_rf.transform([[thresholds['rf']]])
xg_thresh = scaler_xg.transform([[thresholds['xg']]])
ada_thresh = scaler_ada.transform([[thresholds['ada']]])


# 繪圖函數（改為不重疊顯示，標示閾值值）
def plot_distribution(score, y, threshold, model_name, ax, title_suffix='(score)'):
    bins = np.linspace(np.min(score), np.max(score), 40)
    ax.hist(score[y == 0], bins=bins, color='blue', alpha=0.7, label='Survived (label=0)', density=True)
    ax.hist(score[y == 1], bins=bins, color='red', alpha=0.7, label='Dead (label=1)', density=True)
    
    # 如果 threshold 是 array，就轉為 float
    threshold = float(threshold)

    ax.axvline(threshold, color='black', linestyle='--')
    ax.text(threshold, ax.get_ylim()[1]*0.95, f'Threshold = {threshold:.2f}',
            rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=9, color='black')
    
    ax.set_title(f"[{model_name}] {title_suffix}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability Density")
    ax.legend()

# 儲存所有圖進 PDF
pdf_path = "model_distributions.pdf"
with PdfPages(pdf_path) as pdf:
    # 原始 score
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_distribution(ct_scaled, y_test, ct_thresh, "CT", axs[0, 0])
    plot_distribution(rf_scaled, y_test, rf_thresh, "RF", axs[0, 1])
    plot_distribution(xg_scaled, y_test, xg_thresh, "XGBoost", axs[1, 0])
    plot_distribution(ada_scaled, y_test, ada_thresh, "AdaBoost", axs[1, 1])
    fig.suptitle("Model Score Distributions (Original Scaled Scores)", fontsize=16)
    pdf.savefig(fig)
    plt.close()

    # 閾值為 0 的差分分數
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_distribution(ct_pred, y_test, 0, "CT", axs[0, 0], title_suffix="(Score - Threshold)")
    plot_distribution(rf_pred, y_test, 0, "RF", axs[0, 1], title_suffix="(Score - Threshold)")
    plot_distribution(xg_pred, y_test, 0, "XGBoost", axs[1, 0], title_suffix="(Score - Threshold)")
    plot_distribution(ada_pred, y_test, 0, "AdaBoost", axs[1, 1], title_suffix="(Score - Threshold)")
    fig.suptitle("Model Score Distributions (Score - Threshold = 0)", fontsize=16)
    pdf.savefig(fig)
    plt.close()

pdf_path  # 回傳 PDF 檔案路徑
'''