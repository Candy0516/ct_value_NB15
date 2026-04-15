import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


#-------------------------------------------------------------------------------------
#全部特徵

pdf_save_path = 'fig_dash_ct0/feature_distribution_ct_train.pdf'
pdf = PdfPages(pdf_save_path)

path_dir = r'D:\candy\TONIOT'

#result_csv_path = "D:/candy/TONIOT/ctsum_fthreshold_4dataset/ctsum/ct/test_new_features.csv"
#result_csv_path = os.path.join(path_dir, 'data', '4_sum', 'ct-value', 'train.csv')
result_csv_path = os.path.join(path_dir, 'data', '8_dash_ct_0', 'train.csv')

if not os.path.exists(result_csv_path):
    print(f"❌ 檔案不存在: {result_csv_path}")
    exit()

test_results = pd.read_csv(result_csv_path)

# 排除欄位
exclude_cols = ['label']
feature_cols = [c for c in test_results.columns if c not in exclude_cols]

label_0_data = test_results[test_results['label'] == 0]
label_1_data = test_results[test_results['label'] == 1]

for feature in feature_cols:

    fig, ax = plt.subplots(figsize=(6,5))

    ax.hist(
        [label_0_data[feature], label_1_data[feature]],
        bins=40,
        color=['blue','red'],
        alpha=0.7,
        label=['Benign','Attack'],
        density=True
    )

    ax.axvline(x=0, color='black', linestyle='--')

    ax.set_title(f'{feature} Distribution')
    ax.set_xlabel(feature)
    ax.set_ylabel('Proportion')

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

pdf.close()

print(f"\n✅ 所有 feature 分布圖已儲存至 PDF：{pdf_save_path}")


#---------------------------------------------------------------------------------------------------
#所有攻擊
# ===============================
# 設定路徑
# ===============================
# data_dir = r'D:\candy\NB15\CT_attack_cat'
# output_dir = r'4dataset_fig_attackcat'
# os.makedirs(output_dir, exist_ok=True)

# # ===============================
# # 找所有 test 開頭的 CSV
# # ===============================
# files = [f for f in os.listdir(data_dir) if f.startswith("val") and f.endswith(".csv")]

# print("Found files:", files)

# # ===============================
# # 主迴圈
# # ===============================
# for file in files:

#     file_path = os.path.join(data_dir, file)

#     print(f"\nProcessing file: {file}")

#     df = pd.read_csv(file_path)

#     # ===============================
#     # 取得 attack 名稱
#     # ===============================
#     # test_ct_atk_generic.csv → generic
#     name = file.replace(".csv", "")

#     if "atk_" in name:
#         attack_name = name.split("atk_")[-1]
#     else:
#         attack_name = "global"

#     # ===============================
#     # PDF 路徑
#     # ===============================
#     pdf_save_path = os.path.join(
#         output_dir,
#         f'feature_distribution_ct_val_{attack_name}.pdf'
#     )

#     pdf = PdfPages(pdf_save_path)

#     # ===============================
#     # feature
#     # ===============================
#     exclude_cols = ['label', 'attack_cat']
#     feature_cols = [c for c in df.columns if c not in exclude_cols]

#     label_0_data = df[df['label'] == 0]
#     label_1_data = df[df['label'] == 1]

#     # ===============================
#     # 畫圖
#     # ===============================
#     for feature in feature_cols:

#         fig, ax = plt.subplots(figsize=(6, 5))

#         ax.hist(
#             [label_0_data[feature], label_1_data[feature]],
#             bins=40,
#             alpha=0.7,
#             label=['Benign', 'Attack'],
#             density=True
#         )

#         ax.axvline(x=0, linestyle='--')

#         ax.set_title(f'{feature} Distribution ({attack_name})')
#         ax.set_xlabel(feature)
#         ax.set_ylabel('Proportion')

#         ax.legend()
#         ax.grid(True)

#         plt.tight_layout()

#         pdf.savefig(fig)
#         plt.close(fig)

#     pdf.close()

#     print(f"✅ Saved: {pdf_save_path}")

# print("\n🎉 所有檔案完成！")