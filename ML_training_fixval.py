import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import joblib
from models import Get_Model
import numpy as np

#SAMPLE_TYPES = ['none','over','smote','under', 'KMeansSMOTE_binary']
MODEL_TYPES = ['ada','sxgb','usxgb','rf']
#LABELS = ['Dead within 24hr','Dead within 72hr','Dead within 168hr','Finally dead']
#DATA_TYPES = ['ori','ct']
DATA_TYPES = ['ori']

path_dir = r'D:\candy\TONIOT'
all_metrics_list = []

for data_type in DATA_TYPES:
    print(f'Data Type: {data_type}')
    # 迴圈執行每個label
    # for label in LABELS:
    #     print(f'label: {label}\n')
        # 迴圈執行每個model type
    for model_type in MODEL_TYPES:
        print(f'Model Type: {model_type}\n')
            # 迴圈執行每個sample type
            #for sample_type in SAMPLE_TYPES:
        if data_type == 'ori':
            train_dir = os.path.join(path_dir, 'data', '1_preprocess')
            test_dir = os.path.join(path_dir, 'data', '1_preprocess')
            val_dir = os.path.join(path_dir, 'data', '1_preprocess')
            val_name = 'val.csv'
            test_name = 'test.csv'
        elif data_type == 'ct':
            train_dir = os.path.join(path_dir, 'data', '3_DataWithPvalue', 'ct-value')
            #train_dir = os.path.join(path_dir, 'data', '4_sum', 'ct-value')
            #test_dir = os.path.join(path_dir, 'data', '7_sum_test')
            #val_dir = os.path.join(path_dir, 'data', '7_sum_val')
            test_dir = os.path.join(path_dir, 'data', '6_mapped_test')
            val_dir = os.path.join(path_dir, 'data', '6_mapped_val')            
            test_name = 'benign_test.csv'
            val_name = 'benign_val.csv'
        elif data_type == 'ct_fs':
            train_dir = os.path.join(path_dir, 'ctsum_fthreshold_4dataset', 'AUROC90up', 'ctsum', 'ct')
            #train_dir = os.path.join(path_dir, 'data', '4_sum', 'ct-value')
            #test_dir = os.path.join(path_dir, 'data', '7_sum_test')
            #val_dir = os.path.join(path_dir, 'data', '7_sum_val')
            test_dir = os.path.join(path_dir, 'ctsum_fthreshold_4dataset', 'AUROC90up', 'ctsum', 'ct')
            val_dir = os.path.join(path_dir, 'ctsum_fthreshold_4dataset', 'AUROC90up', 'ctsum', 'ct')            
            test_name = 'test_new_features.csv'
            val_name = 'val_new_features.csv'
        else:
            raise NameError('data_type should be \'ct\' or \'ori\'')
        #print(f'Sample Type: {sample_type}')

        # 設置結果保存的資料夾
        #result_dir = os.path.join('result', label, model_type + '_peerj', data_type, sample_type)
        #result_dir = os.path.join('result', model_type, data_type, 'CTAUROC90up')
        result_dir = os.path.join('result', model_type, data_type)
        os.makedirs(result_dir, exist_ok=True)
        
        # 讀取資料
        train = pd.read_csv(os.path.join(train_dir, 'train.csv'))
        #train = pd.read_csv(os.path.join(train_dir, 'train_new_features.csv'))
        test = pd.read_csv(os.path.join(test_dir, test_name))
        val = pd.read_csv(os.path.join(val_dir, val_name))

        #丟特徵--------------------------------
        drop_columns = []

        # 建立 X/y 並丟棄 drop_columns
        X_train = train.drop(columns=['label'] + drop_columns, errors='ignore')
        y_train = train['label']

        X_test = test.drop(columns=['label'] + drop_columns, errors='ignore')
        y_test = test['label']

        X_val = val.drop(columns=['label'] + drop_columns, errors='ignore')
        y_val = val['label']

        #-------------------------------
        # 要保留的特徵
        # keep_columns = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'Dpkts', 'smeansz', 'dmeansz', 'dintpkt', 'ct_state_ttl']

        # # 建立 X / y
        # X_train = train[keep_columns]
        # y_train = train['label']

        # X_test = test[keep_columns]
        # y_test = test['label']

        # X_val = val[keep_columns]
        # y_val = val['label']

        
        print('    Training model')
        model = Get_Model(train, model_type)
        model.fit(X_train, y_train)

        # 保存模型
        model_save_path = os.path.join(result_dir, 'model.pkl')
        joblib.dump(model, model_save_path)

        print('    Predicting on validation set to find best threshold')
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_val)
        youden_index = tpr - fpr
        best_threshold_index = youden_index.argmax()
        best_threshold_val = thresholds[best_threshold_index]

        print('    Predicting on test set')
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred = model.predict(X_test)

        # 計算評估指標
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba_test)

        # 計算 Sensitivity 和 Specificity
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        sensitivity = tp / (tp + fn)  # TPR
        specificity = tn / (tn + fp)  # TNR

        # 計算 AUPRC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
        auprc = auc(recall, precision)

        # 假設 y_test 是真實標籤，y_pred_proba 是預測概率
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)

        # 計算 Youden's J 指數
        youden_index = tpr - fpr
        best_threshold_index = youden_index.argmax()
        best_threshold = thresholds[best_threshold_index]

        # 對應的最佳 TPR 和 TNR
        best_tpr = tpr[best_threshold_index]
        best_tnr = 1 - fpr[best_threshold_index]

        # 計算 PPV 和 NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # 輸出評估結果到 txt 檔案
        with open(os.path.join(result_dir, 'results.txt'), 'w') as f:
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Confusion Matrix:\n{cm}\n')
            f.write(f'Sensitivity: {sensitivity:.4f}\n')
            f.write(f'Specificity: {specificity:.4f}\n')
            f.write(f'PPV (Precision): {ppv:.4f}\n')
            f.write(f'NPV: {npv:.4f}\n')
            f.write(f'AUROC: {auroc:.4f}\n')
            f.write(f'AUPRC: {auprc:.4f}\n')
            f.write(f'best_threshold: {best_threshold:.4f}\n')
            f.write(f'best_tpr: {best_tpr:.4f}\n')
            f.write(f'best_tnr: {best_tnr:.4f}\n')
            f.write(f"Youden's J: {youden_index[best_threshold_index]:.4f}")

        
        # 繪製並保存 ROC 曲線
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')  # 繪製隨機猜測的參考線
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(os.path.join(result_dir, 'roc_curve.png'))  # 保存 ROC 圖片
        plt.close()

        # 繪製並保存 Precision-Recall 曲線
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AUPRC = {auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(os.path.join(result_dir, 'pr_curve.png'))  # 保存 Precision-Recall 圖片
        plt.close()

        # 保存預測機率到 CSV 檔案
        y_pred_proba_df_test = pd.DataFrame({'y_pred_proba_test': y_pred_proba_test, 'y_true_test': y_test})
        y_pred_proba_df_test.to_csv(os.path.join(result_dir, 'y_pred_proba_test.csv'), index=False)
        y_pred_proba_df_train = pd.DataFrame({'y_pred_proba_train': y_pred_proba_train, 'y_true_train': y_train})
        y_pred_proba_df_train.to_csv(os.path.join(result_dir, 'y_pred_proba_train.csv'), index=False)

        print(f'Results saved in {result_dir}.\n')

        #adjust
        # 使用最佳閾值進行測試集預測
        y_pred_adjusted = (y_pred_proba_test >= best_threshold_val).astype(int)

        # 計算調整後的混淆矩陣
        cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
        tp_adjusted = cm_adjusted[1, 1]
        tn_adjusted = cm_adjusted[0, 0]
        fp_adjusted = cm_adjusted[0, 1]
        fn_adjusted = cm_adjusted[1, 0]

        # 調整後評估指標
        accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
        sensitivity_adjusted = tp_adjusted / (tp_adjusted + fn_adjusted) if (tp_adjusted + fn_adjusted) > 0 else 0
        specificity_adjusted = tn_adjusted / (tn_adjusted + fp_adjusted) if (tn_adjusted + fp_adjusted) > 0 else 0
        ppv_adjusted = tp_adjusted / (tp_adjusted + fp_adjusted) if (tp_adjusted + fp_adjusted) > 0 else 0
        npv_adjusted = tn_adjusted / (tn_adjusted + fn_adjusted) if (tn_adjusted + fn_adjusted) > 0 else 0

        # AUROC / AUPRC（無須改名，因為數值一樣，只是繪圖時顯示）
        auroc = roc_auc_score(y_test, y_pred_proba_test)
        precision_adjusted, recall_adjusted, _ = precision_recall_curve(y_test, y_pred_proba_test)
        auprc = auc(recall_adjusted, precision_adjusted)

        # 儲存調整後評估結果
        with open(os.path.join(result_dir, 'adjusted_results.txt'), 'w') as f:
            f.write(f'Adjusted Accuracy: {accuracy_adjusted:.4f}\n')
            f.write(f'Adjusted Sensitivity (TPR): {sensitivity_adjusted:.4f}\n')
            f.write(f'Adjusted Specificity (TNR): {specificity_adjusted:.4f}\n')
            f.write(f'Adjusted PPV (Precision): {ppv_adjusted:.4f}\n')
            f.write(f'Adjusted NPV: {npv_adjusted:.4f}\n')
            f.write(f'Adjusted Confusion Matrix:\n{cm_adjusted}\n')
            f.write(f'Best Threshold (from val set): {best_threshold:.4f}\n')

        # 繪製 ROC 圖
        fpr_adjusted, tpr_adjusted, _ = roc_curve(y_test, y_pred_proba_test)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_adjusted, tpr_adjusted, label=f'Adjusted ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Adjusted ROC Curve')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(os.path.join(result_dir, 'adjusted_roc_curve.png'))
        plt.close()

        # 繪製 Precision-Recall 圖
        plt.figure(figsize=(8, 6))
        plt.plot(recall_adjusted, precision_adjusted, label=f'Adjusted PR curve (AUPRC = {auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Adjusted Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(os.path.join(result_dir, 'adjusted_pr_curve.png'))
        plt.close()

        # 儲存詳細預測結果
        prediction_result_df = pd.DataFrame({
            'y_pred_proba_test': y_pred_proba_test,
            'y_pred': y_pred,
            'y_pred_adjusted': y_pred_adjusted,
            'y_true': y_test,
            'correct': (y_pred == y_test).astype(int),
            'correct_adjusted': (y_pred_adjusted == y_test).astype(int)
        })
        prediction_result_df.to_csv(os.path.join(result_dir, 'prediction_result.csv'), index=False)

        print(f'    Adjusted results saved in {result_dir}.')

        # 平均 TPR + TNR
        balanced_acc = (sensitivity_adjusted + specificity_adjusted) / 2

        # 整理為 DataFrame 一筆
        metrics_df = pd.DataFrame([{
            #'label': label,                      
            'model_type': model_type,           
            #'sample_type': sample_type,
            'Original Accuracy': accuracy * 100,         
            'Accuracy': accuracy_adjusted * 100,
            'AUROC': auroc * 100,
            'AUPRC': auprc * 100,
            'Original TPR': sensitivity * 100,
            'Original TNR': specificity * 100,
            'Adjusted TPR': sensitivity_adjusted * 100,
            'Adjusted TNR': specificity_adjusted * 100,
            'Balanced Accuracy': balanced_acc * 100,
            'Adjusted PPV': ppv_adjusted * 100,
            'Adjusted NPV': npv_adjusted * 100,
            'best_tpr': best_tpr * 100,
            'best_tnr': best_tnr * 100
        }])

        all_metrics_list.append(metrics_df)

        '''
        # 儲存或 append 至 CSV
        csv_path = os.path.join(result_dir, "summary_metrics.csv")
        if os.path.exists(csv_path):
            prev_df = pd.read_csv(csv_path)
            metrics_df = pd.concat([prev_df, metrics_df], ignore_index=True)
        metrics_df.to_csv(csv_path, index=False)
        print(f"✅ 指標已儲存：{csv_path}")
        '''

if all_metrics_list:
    all_metrics_df = pd.concat(all_metrics_list, ignore_index=True)
    all_metrics_df.to_csv(os.path.join(path_dir, 'all_metrics_summary_rcsmote.csv'), index=False)
    print("📊 所有指標已統整儲存：all_metrics_summary2.csv")


