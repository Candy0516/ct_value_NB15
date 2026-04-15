import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import gc

# ==============================
# 檔案路徑
# ==============================

orig_csv = "D:/candy/NB15/data/1_preprocess/val.csv"
ct_csv   = "D:/candy/NB15/data/6_mapped_val/benign_val.csv"

pdf_path = "4dataset_fig_no_ratio/ct_cross_ori_distribution_val_bin100_proportion_v2.pdf"

# ==============================
# 讀取資料
# ==============================

orig_df = pd.read_csv(orig_csv)
ct_df   = pd.read_csv(ct_csv)

label_col = "label"

feature_cols = [c for c in orig_df.columns if c != label_col]

# ==============================
# PDF
# ==============================

pdf = PdfPages(pdf_path)

#1.直方圖--------------------------------------------------------------------
# ==============================
# 每個特徵處理
# ==============================

# for feature in feature_cols:

#     print("Processing:", feature)

#     original_values = orig_df[feature]   # X軸
#     ct_values = ct_df[feature]           # 判斷越界
#     labels = ct_df[label_col]

#     # ==========================
#     # 越界條件
#     # ==========================

#     cross_mask = (
#         ((ct_values < 0) & (labels == 0)) |
#         ((ct_values > 0) & (labels == 1))
#     )

#     non_cross_mask = ~cross_mask

#     # ==========================
#     # 分 label
#     # ==========================

#     benign_cross = original_values[(labels == 0) & cross_mask]
#     attack_cross = original_values[(labels == 1) & cross_mask]
#     benign_non = original_values[(labels == 0) & non_cross_mask]
#     attack_non = original_values[(labels == 1) & non_cross_mask]

#     # ==========================
#     # 畫圖
#     # ==========================

#     fig, axes = plt.subplots(2,1, figsize=(10,8))

#     # --------------------------
#     # 上圖：越界
#     # --------------------------

#     axes[0].hist(
#         benign_cross,
#         bins=40,
#         alpha=0.6,
#         color='blue',
#         density=True,
#         label='Benign'
#     )

#     axes[0].hist(
#         attack_cross,
#         bins=40,
#         alpha=0.6,
#         color='red',
#         density=True,
#         label='Attack'
#     )

#     axes[0].axvline(x=0, linestyle='--', color='black')
#     axes[0].set_title(f"{feature} Cross Distribution")
#     axes[0].set_xlabel("Original Value")
#     axes[0].set_ylabel("Proportion")
#     axes[0].legend()
#     axes[0].grid(True)

#     # --------------------------
#     # 下圖：沒越界
#     # --------------------------

#     axes[1].hist(
#         benign_non,
#         bins=40,
#         alpha=0.6,
#         color='blue',
#         density=True,
#         label='Benign'
#     )

#     axes[1].hist(
#         attack_non,
#         bins=40,
#         alpha=0.6,
#         color='red',
#         density=True,
#         label='Attack'
#     )

#     axes[1].axvline(x=0, linestyle='--', color='black')
#     axes[1].set_title(f"{feature} Non-Cross Distribution")
#     axes[1].set_xlabel("Original Value")
#     axes[1].set_ylabel("Proportion")
#     axes[1].legend()
#     axes[1].grid(True)

#     plt.tight_layout()

#     pdf.savefig(fig)
#     plt.close()

# pdf.close()

# print("PDF saved:", pdf_path)

#2.----------------------------------------------------------------------------------------------------------
#bin=100 count/proportion
'''
for feature in feature_cols:

    print("Processing:", feature)

    original_values = orig_df[feature].dropna()
    ct_values = ct_df[feature]
    labels = ct_df[label_col]

    cross_mask = (
        ((ct_values < 0) & (labels == 0)) |
        ((ct_values > 0) & (labels == 1))
    )

    non_cross_mask = ~cross_mask

    benign_cross = orig_df[feature][(labels==0) & cross_mask].dropna()
    attack_cross = orig_df[feature][(labels==1) & cross_mask].dropna()

    benign_non = orig_df[feature][(labels==0) & non_cross_mask].dropna()
    attack_non = orig_df[feature][(labels==1) & non_cross_mask].dropna()

    # ===== bin數自動決定 =====
    unique_vals = original_values.nunique()

    if unique_vals > 100:
        bins = 100
    else:
        bins = unique_vals

    # ===== histogram計算 =====
    #count
    # hist_bc, edges = np.histogram(benign_cross, bins=bins)
    # hist_ac, _ = np.histogram(attack_cross, bins=edges)

    # hist_bn, edges2 = np.histogram(benign_non, bins=bins)
    # hist_an, _ = np.histogram(attack_non, bins=edges2)

    #proportion
    hist_bc, edges = np.histogram(benign_cross, bins=bins)
    hist_ac, _ = np.histogram(attack_cross, bins=edges)

    # 轉比例
    if len(benign_cross) > 0:
        hist_bc = hist_bc / len(benign_cross) *100

    if len(attack_cross) > 0:
        hist_ac = hist_ac / len(attack_cross) *100
    
    hist_bn, edges2 = np.histogram(benign_non, bins=bins)
    hist_an, _ = np.histogram(attack_non, bins=edges2)

    if len(benign_non) > 0:
        hist_bn = hist_bn / len(benign_non) *100

    if len(attack_non) > 0:
        hist_an = hist_an / len(attack_non) *100
    #-----------------------------------------------------

    centers = (edges[:-1] + edges[1:]) / 2
    centers2 = (edges2[:-1] + edges2[1:]) / 2

    fig, axes = plt.subplots(2,1, figsize=(10,8))

    # ===== Cross =====
    axes[0].bar(centers, hist_bc, width=centers[1]-centers[0],
                color="blue", alpha=0.6, label="Benign")

    axes[0].bar(centers, hist_ac, width=centers[1]-centers[0],
                color="red", alpha=0.6, label="Attack")

    axes[0].axvline(0, linestyle="--", color="black")

    axes[0].set_title(f"{feature} Cross Distribution")
    axes[0].set_xlabel("Original Value")
    #axes[0].set_ylabel("Count")
    axes[0].set_ylabel("Proportion")
    axes[0].legend()
    axes[0].grid(True)

    # ===== Non Cross =====
    axes[1].bar(centers2, hist_bn, width=centers2[1]-centers2[0],
                color="blue", alpha=0.6, label="Benign")

    axes[1].bar(centers2, hist_an, width=centers2[1]-centers2[0],
                color="red", alpha=0.6, label="Attack")

    axes[1].axvline(0, linestyle="--", color="black")

    axes[1].set_title(f"{feature} Non-Cross Distribution")
    axes[1].set_xlabel("Original Value")
    #axes[1].set_ylabel("Count")
    axes[1].set_ylabel("Proportion")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close()

pdf.close()

print("PDF saved:", pdf_path)
'''

#3.------------------------------------------------------------------------------------------------------------------
#另一種比例
for feature in feature_cols:

    print("Processing:", feature)

    original_values = orig_df[feature].dropna()
    ct_values = ct_df[feature]
    labels = ct_df[label_col]

    cross_mask = (
        ((ct_values < 0) & (labels == 0)) |
        ((ct_values > 0) & (labels == 1))
    )

    non_cross_mask = ~cross_mask

    benign_cross = orig_df[feature][(labels==0) & cross_mask].dropna()
    attack_cross = orig_df[feature][(labels==1) & cross_mask].dropna()

    benign_non = orig_df[feature][(labels==0) & non_cross_mask].dropna()
    attack_non = orig_df[feature][(labels==1) & non_cross_mask].dropna()

    # ===== bin數自動決定 =====
    unique_vals = original_values.nunique()

    if unique_vals > 100:
        bins = 100
    else:
        bins = unique_vals

    #proportion
    # 所有資料 (當作分母)
    hist_total, edges = np.histogram(original_values, bins=bins)

    # cross
    hist_bc, _ = np.histogram(benign_cross, bins=edges)
    hist_ac, _ = np.histogram(attack_cross, bins=edges)

    # non-cross
    hist_bn, _ = np.histogram(benign_non, bins=edges)
    hist_an, _ = np.histogram(attack_non, bins=edges)

    # 避免 division by zero
    hist_total_safe = np.where(hist_total == 0, 1, hist_total)

    # 轉比例 (%)
    hist_bc = hist_bc / hist_total_safe * 100
    hist_ac = hist_ac / hist_total_safe * 100
    hist_bn = hist_bn / hist_total_safe * 100
    hist_an = hist_an / hist_total_safe * 100
    #-----------------------------------------------------

    centers = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(2,1, figsize=(10,8))

    # ===== Cross =====
    axes[0].bar(centers, hist_bc, width=centers[1]-centers[0],
                color="blue", alpha=0.6, label="Benign")

    axes[0].bar(centers, hist_ac, width=centers[1]-centers[0],
                color="red", alpha=0.6, label="Attack")

    axes[0].axvline(0, linestyle="--", color="black")

    axes[0].set_title(f"{feature} Cross Distribution")
    axes[0].set_xlabel("Original Value")
    #axes[0].set_ylabel("Count")
    axes[0].set_ylabel("Proportion")
    axes[0].legend()
    axes[0].grid(True)

    # ===== Non Cross =====
    axes[1].bar(centers, hist_bn, width=centers[1]-centers[0],
                color="blue", alpha=0.6, label="Benign")

    axes[1].bar(centers, hist_an, width=centers[1]-centers[0],
                color="red", alpha=0.6, label="Attack")

    axes[1].axvline(0, linestyle="--", color="black")

    axes[1].set_title(f"{feature} Non-Cross Distribution")
    axes[1].set_xlabel("Original Value")
    #axes[1].set_ylabel("Count")
    axes[1].set_ylabel("Proportion")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close()

pdf.close()

print("PDF saved:", pdf_path)