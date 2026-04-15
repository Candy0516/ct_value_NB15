import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)

#SAMPLE_TYPES = ['none', 'over', 'smote', 'under', 'KMeansSMOTE_binary', 'classweight']
#LABELS = ['Dead within 24hr', 'Dead within 72hr', 'Dead within 168hr', 'Finally dead']

model_type = 'ctsum'
data_type = 'ct'
path_dir = r'D:\candy\NB15'
all_metrics_list = []

# for label in LABELS:
#     print(f'label: {label}\n')
#     label_auroc = []
#     label_adj_mean = []
#     for sample_type in SAMPLE_TYPES:
#         print(f'Sample Type: {sample_type}')
result_dir = os.path.join(path_dir, 'result_split4dataset_fsshap', model_type, data_type)
os.makedirs(result_dir, exist_ok=True)

# === 資料讀取 ===
test = pd.read_csv(os.path.join(path_dir, 'data', '7_sum_test', 'benign_test.csv'))
train = pd.read_csv(os.path.join(path_dir, 'data', '4_sum', 'ct-value', 'train.csv'))
val = pd.read_csv(os.path.join(path_dir, 'data', '7_sum_val', 'benign_val.csv'))

shap_stats_path = "D:/candy/NB15/result_split_4dataset_shap/ctsum/ct/ctsum_shap_feature_statistics.csv"
shap_stats_df = pd.read_csv(shap_stats_path)
# === 驗證集找最佳 threshold ===
'''
y_val = val['label']
y_pred_proba_val = -val['sum']
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_proba_val)
j_scores = tpr_val - fpr_val
best_threshold_index = j_scores.argmax()
best_threshold_val = thresholds_val[best_threshold_index]
'''

# === 被丟棄的特徵 ===
# drop_features = [
#     'dwin', 'swin', 'is_ftp_login', 'trans_depth', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'service', 
#     'proto', 'dintpkt', 'state', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'sloss', 'ct_srv_src'
# ]

# 規則：mean_shap < 0.01 就丟棄
drop_features = shap_stats_df.loc[
    shap_stats_df["mean_shap"] < 0.01, "feature"
].tolist()

print("Drop features (mean_shap < 0.01):")
print(drop_features)
print(f"Total dropped: {len(drop_features)}")

# === 取得剩下要用的特徵 ===
sum_features = [
    c for c in train.columns
    if c not in drop_features and c != 'label'
]

# === 重新計算 sum ===
for df in [train, val, test]:
    df['sum_filtered'] = df[sum_features].sum(axis=1)

# === 驗證集找最佳 threshold ===
y_val = val['label']
y_pred_proba_val = -val['sum_filtered']

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_proba_val)
j_scores = tpr_val - fpr_val
best_threshold_val = thresholds_val[j_scores.argmax()]

# === 測試集預測 ===
y_test = test['label']
y_pred_proba_test = -test['sum_filtered']

y_pred_test = (y_pred_proba_test >= 0).astype(int)
y_pred_adjusted = (y_pred_proba_test >= best_threshold_val).astype(int)

# === 測試集預測 ===
'''
y_test = test['label']
y_pred_proba_test = -test['sum']
y_pred_test = (y_pred_proba_test >= 0).astype(int)

y_pred_adjusted = (y_pred_proba_test >= best_threshold_val).astype(int)
'''

# === 評估指標計算 ===
def get_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = auc(recall, precision)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return acc, auc_score, auprc, tpr, tnr, ppv, npv, cm

acc0, auroc0, auprc0, tpr0, tnr0, ppv0, npv0, cm0 = get_metrics(y_test, y_pred_test, y_pred_proba_test)
# 計算 Youden's J 指數
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
youden_index = tpr - fpr
best_threshold_index = youden_index.argmax()
best_threshold = thresholds[best_threshold_index]

# 對應的最佳 TPR 和 TNR
best_tpr = tpr[best_threshold_index]
best_tnr = 1 - fpr[best_threshold_index]

acc1, auroc1, auprc1, tpr1, tnr1, ppv1, npv1, cm1 = get_metrics(y_test, y_pred_adjusted, y_pred_proba_test)

# === 儲存 txt/CSV 結果 ===
with open(os.path.join(result_dir, 'results.txt'), 'w') as f:
    f.write(f'Accuracy: {acc0:.4f}\n')
    f.write(f'Confusion Matrix:\n{cm0}\n')
    f.write(f'Sensitivity: {tpr0:.4f}\n')
    f.write(f'Specificity: {tnr0:.4f}\n')
    f.write(f'PPV (Precision): {ppv0:.4f}\n')
    f.write(f'NPV: {npv0:.4f}\n')
    f.write(f'AUROC: {auroc0:.4f}\n')
    f.write(f'AUPRC: {auprc0:.4f}\n')
    f.write(f'best_threshold: {best_threshold:.4f}\n')
    f.write(f'best_tpr: {best_tpr:.4f}\n')
    f.write(f'best_tnr: {best_tnr:.4f}\n')
    f.write(f"Youden's J: {youden_index[best_threshold_index]:.4f}")

with open(os.path.join(result_dir, 'adjusted_results.txt'), 'w') as f:
    f.write(f'Adjusted Accuracy: {acc1:.4f}\n')
    f.write(f'Adjusted Confusion Matrix:\n{cm1}\n')
    f.write(f'Adjusted TPR: {tpr1:.4f}\n')
    f.write(f'Adjusted TNR: {tnr1:.4f}\n')
    f.write(f'Adjusted PPV: {ppv1:.4f}\n')
    f.write(f'Adjusted NPV: {npv1:.4f}\n')
    f.write(f'AUROC: {auroc1:.4f}\n')
    f.write(f'AUPRC: {auprc1:.4f}\n')


pd.DataFrame({
    'sum': test['sum'],
    'y_pred': y_pred_test,
    'y_pred_adjusted': y_pred_adjusted,
    'y_true': y_test,
    'correct': (y_pred_test == y_test).astype(int),
    'correct_adjusted': (y_pred_adjusted == y_test).astype(int)
}).to_csv(os.path.join(result_dir, 'prediction_result.csv'), index=False)

metrics_dict = {
    #'Label': label,
    'model_type': model_type,
    #'sample_type': sample_type,
    'Accuracy': acc1 * 100,
    'AUROC': auroc0 * 100,
    'AUPRC': auprc0 * 100,
    'Original TPR': tpr0 * 100,
    'Original TNR': tnr0 * 100,
    'Adjusted TPR': tpr1 * 100,
    'Adjusted TNR': tnr1 * 100,
    'Balanced Accuracy': (tpr1 + tnr1) * 50,
    'Adjusted PPV': ppv1 * 100,
    'Adjusted NPV': npv1 * 100,
    'best_tpr': best_tpr * 100,
    'best_tnr': best_tnr * 100
}

pd.DataFrame([metrics_dict]).to_csv(os.path.join(result_dir, 'summary_metrics.csv'), index=False)
all_metrics_list.append(metrics_dict)

# === 繪製 ROC Curve（使用測試集預測分數與真實標籤） ===
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc0:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')  # 隨機參考線
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'ROC Curve  ')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
plt.close()

# === 繪製 Precision-Recall Curve（使用測試集預測分數與真實標籤） ===
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (AUPRC = {auprc0:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve ')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'pr_curve.png'))
plt.close()

# === 儲存總表 ===
all_metrics_df = pd.DataFrame(all_metrics_list)
all_metrics_df.to_csv('all_metrics_summary_ctcw_split_4dataset_fs_shap.csv', index=False)
print("✅ 統整與圖表已完成！")