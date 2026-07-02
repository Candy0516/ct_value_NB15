import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve,
    precision_score, recall_score, f1_score
)
model_type = 'ctsum'
data_type = 'ct'
path_dir = r'D:\candy\NB15'
result_dir = os.path.join(path_dir, 'compare', 'alldata_fs', 'AUROC_fs', 'AUROC80up_alltest', model_type, data_type)
os.makedirs(result_dir, exist_ok=True)

# ==========================
# 讀取資料
# ==========================

train = pd.read_csv(os.path.join(path_dir,'data','4_sum','ct-value','train.csv'))
val   = pd.read_csv(os.path.join(path_dir,'data','7_sum_val','benign_val.csv'))
test  = pd.read_csv(os.path.join(path_dir,'data','7_sum_test','benign_test.csv'))
#test  = pd.read_csv('D:/candy/NB15/data/ct_value_modified_4dataset.csv')

feature_cols = [c for c in train.columns if c not in ['label']]

def get_metrics(y_true, y_pred, y_proba):

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    auc_score = roc_auc_score(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = auc(recall, precision)

    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp+fn) if (tp+fn)>0 else 0
    tnr = tn / (tn+fp) if (tn+fp)>0 else 0
    ppv = tp / (tp+fp) if (tp+fp)>0 else 0
    npv = tn / (tn+fn) if (tn+fn)>0 else 0

    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1_val = f1_score(y_true, y_pred, zero_division=0)

    return acc, auc_score, auprc, tpr, tnr, ppv, npv, precision_val, recall_val, f1_val, cm

# ==========================
# 計算每個 feature threshold
# ==========================

threshold_dict = {}
feature_metrics = []

feature_cols = [c for c in train.columns if c != 'label']

for feature in feature_cols:

    print(f'Processing feature: {feature}')

    # ==========================
    # Validation 找最佳 threshold
    # ==========================
    y_val = val['label']
    y_val_score = -val[feature]

    val_auroc = roc_auc_score(y_val, y_val_score)

    fpr, tpr, thresholds = roc_curve(y_val, y_val_score)

    youden_index = tpr - fpr
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]

    threshold_dict[feature] = best_threshold

    # ==========================
    # Test prediction
    # ==========================
    y_test = test['label']
    y_test_score = -test[feature]

    y_pred = (y_test_score >= best_threshold).astype(int)

    acc, test_auroc, auprc, tpr1, tnr1, ppv, npv, precision_val, recall_val, f1_val, cm = get_metrics(
        y_test,
        y_pred,
        y_test_score
    )

    # ==========================
    # 紀錄結果
    # ==========================
    feature_metrics.append({
        'feature': feature,
        'threshold': best_threshold,
        'Val_AUROC': val_auroc,
        'Test_AUROC': test_auroc,
        'Accuracy': acc,
        'AUPRC': auprc,
        'Precision': precision_val,
        'Recall': recall_val,
        'F1': f1_val,
        'TPR': tpr1,
        'TNR': tnr1,
        'PPV': ppv,
        'NPV': npv
    })

# ==========================
# 輸出 CSV
# ==========================

feature_metrics_df = pd.DataFrame(feature_metrics)

feature_metrics_df.to_csv(
    os.path.join(result_dir, 'feature_threshold_metrics.csv'),
    index=False
)

print("✅ Feature threshold + metrics 已輸出")

# ==============================
# Step 2: 使用驗證集 Val_AUROC >= 0.8 特徵選擇
# 並依 Val_AUROC 由大到小排序
# ==============================

AUROC_THRESHOLD = 0.8

selected_feature_df = feature_metrics_df[
    (feature_metrics_df['Val_AUROC'] >= AUROC_THRESHOLD) &
    (feature_metrics_df['feature'] != 'sum')
].copy()

selected_feature_df = selected_feature_df.sort_values(
    by='Val_AUROC',
    ascending=False
).reset_index(drop=True)

selected_features_sorted = selected_feature_df['feature'].tolist()

print("Selected features:", selected_features_sorted)

if len(selected_features_sorted) == 0:
    raise ValueError(f"沒有任何特徵符合 Val_AUROC >= {AUROC_THRESHOLD}")

# 儲存排序後的特徵清單
selected_feature_df.to_csv(
    os.path.join(result_dir, 'selected_features_sorted_by_val_auroc.csv'),
    index=False,
    encoding='utf-8-sig'
)

#使用RF特徵選擇-------------------------------------------------------------
# selected_features_sorted = ['sttl', 'ct_state_ttl', 'rate', 'sload', 'dload', 'dttl', 'sbytes', 'ct_srv_dst', 
#                             'smean', 'dmean', 'dbytes', 'ct_dst_src_ltm', 'tcprtt', 'ackdat', 'synack', 'ct_srv_src', 
#                             'dur', 'state', 'ct_dst_sport_ltm', 'sinpkt']


# ==============================
# Step 3: 定義轉換函式
# ==============================

def transform_features(df, threshold_dict):
    new_df = df.copy()

    for f in threshold_dict:
        new_df[f] = df[f] - threshold_dict[f]

    return new_df


# ==============================
# Step 4: 逐一加入特徵測試
# 第一次 top 1
# 第二次 top 2
# 第三次 top 3 ...
# ==============================

comparison_rows = []

for k in range(1, len(selected_features_sorted) + 1):

    current_features = selected_features_sorted[:k]

    print("=" * 80)
    print(f"Testing top {k} features")
    print("Current features:", current_features)

    # 每次測試建立子資料夾
    sub_result_dir = os.path.join(result_dir, f"top_{k:02d}_features")
    os.makedirs(sub_result_dir, exist_ok=True)

    # 目前使用的 feature thresholds
    selected_thresholds = {
        f: threshold_dict[f] for f in current_features
    }

    # ==============================
    # Step 4.1: 計算新 feature value
    # ==============================
    train_new = transform_features(train, selected_thresholds)
    val_new   = transform_features(val, selected_thresholds)
    test_new  = transform_features(test, selected_thresholds)

    # ==============================
    # Step 4.2: 計算 sum
    # ==============================
    train_new['sum'] = train_new[current_features].sum(axis=1)
    val_new['sum']   = val_new[current_features].sum(axis=1)
    test_new['sum']  = test_new[current_features].sum(axis=1)

    # ==============================
    # Step 4.3: Baseline evaluation threshold = 0
    # ==============================
    y_test = test_new['label']
    score_test = -test_new['sum']

    y_pred_baseline = (score_test >= 0).astype(int)

    acc0, auroc0, auprc0, tpr0, tnr0, ppv0, npv0, precision0, recall0, f1_0, cm0 = get_metrics(
        y_test,
        y_pred_baseline,
        score_test
    )

    # ==============================
    # Step 4.4: validation 找 sum threshold
    # ==============================
    y_val = val_new['label']
    score_val = -val_new['sum']

    fpr, tpr, thresholds = roc_curve(y_val, score_val)

    youden = tpr - fpr
    best_idx = youden.argmax()
    best_sum_threshold = thresholds[best_idx]

    # ==============================
    # Step 4.5: test prediction with best threshold
    # ==============================
    y_pred = (score_test >= best_sum_threshold).astype(int)

    acc, auroc, auprc, tpr1, tnr1, ppv, npv, precision1, recall1, f1_1, cm = get_metrics(
        y_test,
        y_pred,
        score_test
    )

    # ==============================
    # Step 4.6: 輸出該 k 特徵數量的檔案
    # ==============================

    # 儲存目前使用的特徵
    pd.DataFrame({
        'rank': list(range(1, k + 1)),
        'feature': current_features,
        'Val_AUROC': [
            selected_feature_df.loc[
                selected_feature_df['feature'] == f,
                'Val_AUROC'
            ].values[0]
            for f in current_features
        ],
        'threshold': [
            threshold_dict[f]
            for f in current_features
        ]
    }).to_csv(
        os.path.join(sub_result_dir, 'used_features.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # ==============================
    # 只儲存選到的特徵 + sum + label
    # ==============================

    save_cols = current_features + ['sum', 'label']

    train_new_save = train_new[save_cols].copy()
    val_new_save   = val_new[save_cols].copy()
    test_new_save  = test_new[save_cols].copy()

    # 新 feature
    train_new.to_csv(
        os.path.join(sub_result_dir, 'train_new_features.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    val_new.to_csv(
        os.path.join(sub_result_dir, 'val_new_features.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    test_new.to_csv(
        os.path.join(sub_result_dir, 'test_new_features.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # prediction result
    pd.DataFrame({
        'sum': test_new['sum'],
        'y_true': y_test,
        'y_pred_baseline': y_pred_baseline,
        'y_pred_best': y_pred
    }).to_csv(
        os.path.join(sub_result_dir, 'prediction_result.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # metrics
    metrics_df = pd.DataFrame([
        {
            'type': 'baseline',
            'feature_count': k,
            'used_features': '|'.join(current_features),
            'Accuracy': acc0,
            'AUROC': auroc0,
            'AUPRC': auprc0,
            'Precision': precision0,
            'Recall': recall0,
            'F1': f1_0,
            'TPR': tpr0,
            'TNR': tnr0,
            'PPV': ppv0,
            'NPV': npv0,
            'threshold': 0
        },
        {
            'type': 'best_threshold',
            'feature_count': k,
            'used_features': '|'.join(current_features),
            'Accuracy': acc,
            'AUROC': auroc,
            'AUPRC': auprc,
            'Precision': precision1,
            'Recall': recall1,
            'F1': f1_1,
            'TPR': tpr1,
            'TNR': tnr1,
            'PPV': ppv,
            'NPV': npv,
            'threshold': best_sum_threshold
        }
    ])

    metrics_df.to_csv(
        os.path.join(sub_result_dir, 'metrics.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # confusion matrix
    pd.DataFrame(cm0).to_csv(
        os.path.join(sub_result_dir, 'confusion_matrix_baseline.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    pd.DataFrame(cm).to_csv(
        os.path.join(sub_result_dir, 'confusion_matrix_best_threshold.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # ==============================
    # Step 4.7: ROC Curve
    # ==============================
    fpr_test, tpr_test, _ = roc_curve(y_test, score_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, label=f'ROC Curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve CT-sum Top {k} Features')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(sub_result_dir, 'roc_curve.png'))
    plt.close()

    # ==============================
    # Step 4.8: 紀錄總比較
    # ==============================
    comparison_rows.append({
        'feature_count': k,
        'used_features': '|'.join(current_features),

        'Baseline_Accuracy': acc0,
        'Baseline_F1': f1_0,
        'Baseline_Precision': precision0,
        'Baseline_Recall': recall0,
        'Baseline_AUROC': auroc0,
        'Baseline_AUPRC': auprc0,
        'Baseline_TPR': tpr0,
        'Baseline_TNR': tnr0,

        'Best_Accuracy': acc,
        'Best_F1': f1_1,
        'Best_Precision': precision1,
        'Best_Recall': recall1,
        'Best_AUROC': auroc,
        'Best_AUPRC': auprc,
        'Best_TPR': tpr1,
        'Best_TNR': tnr1,
        'Best_threshold': best_sum_threshold,

        'result_folder': sub_result_dir
    })

    print(
        f"Top {k} done | "
        f"Best Accuracy={acc:.6f}, "
        f"Best F1={f1_1:.6f}, "
        f"Best AUROC={auroc:.6f}"
    )


# ==============================
# Step 5: 輸出總比較 CSV
# ==============================

comparison_df = pd.DataFrame(comparison_rows)

comparison_csv_path = os.path.join(
    result_dir,
    'feature_count_comparison_acc_f1.csv'
)

comparison_df.to_csv(
    comparison_csv_path,
    index=False,
    encoding='utf-8-sig'
)

print("=" * 80)
print("✅ 完成逐步加入特徵測試")
print("總比較 CSV 已輸出：", comparison_csv_path)

print("\nBest by Accuracy:")
print(
    comparison_df.sort_values(
        by='Best_Accuracy',
        ascending=False
    ).head(10)[[
        'feature_count',
        'Best_Accuracy',
        'Best_F1',
        'used_features'
    ]]
)

print("\nBest by F1:")
print(
    comparison_df.sort_values(
        by='Best_F1',
        ascending=False
    ).head(10)[[
        'feature_count',
        'Best_Accuracy',
        'Best_F1',
        'used_features'
    ]]
)