import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
model_type = 'ctsum'
data_type = 'ct'
path_dir = r'D:\candy\NB15'
result_dir = os.path.join(path_dir, 'ctsum_fthreshold_4dataset','modity_AUROC90up', model_type, data_type)
os.makedirs(result_dir, exist_ok=True)

# ==========================
# 讀取資料
# ==========================

train = pd.read_csv(os.path.join(path_dir,'data','4_sum','ct-value','train.csv'))
val   = pd.read_csv(os.path.join(path_dir,'data','7_sum_val','benign_val.csv'))
#test  = pd.read_csv(os.path.join(path_dir,'data','7_sum_test','benign_test.csv'))
test  = pd.read_csv('D:/candy/NB15/data/ct_value_modified_4dataset.csv')

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

    return acc, auc_score, auprc, tpr, tnr, ppv, npv, cm

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

    acc, auroc, auprc, tpr1, tnr1, ppv, npv, cm = get_metrics(
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
        'Accuracy': acc,
        'AUROC': auroc,
        'AUPRC': auprc,
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
# Step 2: 選擇 AUROC >= 0.8 特徵
# ==============================

selected_features = feature_metrics_df[
    feature_metrics_df['AUROC'] >= 0.9
]['feature'].tolist()
#selected_features = ['sbytes', 'dbytes', 'sttl', 'dttl', 'Dpkts', 'smeansz', 'dmeansz', 'ct_state_ttl', 'sum']
print("Selected features:", selected_features)

# ==============================
# Step 3: 重新建立 threshold dict
# ==============================

selected_thresholds = {
    f: threshold_dict[f] for f in selected_features
}

# ==============================
# Step 4: 計算新 feature value
# ==============================

def transform_features(df, threshold_dict):

    new_df = df.copy()

    for f in threshold_dict:
        new_df[f] = df[f] - threshold_dict[f]

    return new_df


train_new = transform_features(train, selected_thresholds)
val_new   = transform_features(val, selected_thresholds)
test_new  = transform_features(test, selected_thresholds)

# ==============================
# Step 5: 計算 sum
# ==============================

train_new['sum'] = train_new[selected_features].sum(axis=1)
val_new['sum']   = val_new[selected_features].sum(axis=1)
test_new['sum']  = test_new[selected_features].sum(axis=1)

# ==============================
# Step 5.5: Baseline evaluation (threshold = 0)
# ==============================

y_test = test_new['label']
score_test = -test_new['sum']

y_pred_baseline = (score_test >= 0).astype(int)

acc0, auroc0, auprc0, tpr0, tnr0, ppv0, npv0, cm0 = get_metrics(
    y_test,
    y_pred_baseline,
    score_test
)

baseline_metrics = {
    'Accuracy': acc0,
    'AUROC': auroc0,
    'AUPRC': auprc0,
    'TPR': tpr0,
    'TNR': tnr0,
    'PPV': ppv0,
    'NPV': npv0,
    'threshold': 0
}

# ==============================
# Step 6: validation 找 sum threshold
# ==============================

y_val = val_new['label']
score_val = -val_new['sum']

fpr, tpr, thresholds = roc_curve(y_val, score_val)

youden = tpr - fpr
best_idx = youden.argmax()
best_sum_threshold = thresholds[best_idx]

# ==============================
# Step 7: test prediction (best threshold)
# ==============================

y_pred = (score_test >= best_sum_threshold).astype(int)

acc, auroc, auprc, tpr1, tnr1, ppv, npv, cm = get_metrics(
    y_test,
    y_pred,
    score_test
)

# ==============================
# Step 8: 輸出 CSV
# ==============================

# 新 feature
train_new.to_csv(os.path.join(result_dir,'train_new_features.csv'),index=False)
val_new.to_csv(os.path.join(result_dir,'val_new_features.csv'),index=False)
test_new.to_csv(os.path.join(result_dir,'test_new_features.csv'),index=False)

# prediction
pd.DataFrame({
    'sum': test_new['sum'],
    'y_true': y_test,
    'y_pred_baseline': y_pred_baseline,
    'y_pred_best': y_pred
}).to_csv(
    os.path.join(result_dir, 'prediction_result.csv'),
    index=False
)

# metrics
metrics_df = pd.DataFrame([
{
    'type': 'baseline',
    'Accuracy': acc0,
    'AUROC': auroc0,
    'AUPRC': auprc0,
    'TPR': tpr0,
    'TNR': tnr0,
    'PPV': ppv0,
    'NPV': npv0,
    'threshold': 0
},
{
    'type': 'best_threshold',
    'Accuracy': acc,
    'AUROC': auroc,
    'AUPRC': auprc,
    'TPR': tpr1,
    'TNR': tnr1,
    'PPV': ppv,
    'NPV': npv,
    'threshold': best_sum_threshold
}
])

metrics_df.to_csv(
    os.path.join(result_dir, 'metrics.csv'),
    index=False
)

print("✅ 完成 AUROC feature selection + CT-sum + baseline comparison")

# ==============================
# Step 9: 繪製 ROC Curve
# ==============================

fpr, tpr, _ = roc_curve(y_test, score_test)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.4f})')

# random classifier
plt.plot([0,1],[0,1],'k--',label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (CT-sum)')
plt.legend(loc='best')
plt.grid()

plt.tight_layout()

plt.savefig(os.path.join(result_dir,'roc_curve.png'))

plt.close()

print("ROC curve saved")
