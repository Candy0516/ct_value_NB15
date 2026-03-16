import pandas as pd
import numpy as np

orig_csv = r"D:/candy/NB15/data/1_preprocess/test.csv"
ct_csv   = r"D:/candy/NB15/data/6_mapped_test/benign_test.csv"

new_ct_csv = r"D:/candy/NB15/data/ct_value_modified_4dataset.csv"
report_csv = r"D:/candy/NB15/data/ct_cross_report_4dataset.csv"

label_col = "label"

orig_df = pd.read_csv(orig_csv)
ct_df   = pd.read_csv(ct_csv)

feature_cols = [c for c in orig_df.columns if c != label_col]

report_rows = []

for feature in feature_cols:

    print("Processing:", feature)

    original_values = orig_df[feature]
    ct_values = ct_df[feature]
    labels = ct_df[label_col]

    # 越界條件
    cross_mask = (
        ((ct_values < 0) & (labels == 0)) |
        ((ct_values > 0) & (labels == 1))
    )

    df = pd.DataFrame({
        "orig": original_values,
        "ct": ct_values,
        "cross": cross_mask
    })

    # group by 原始值
    grouped = df.groupby("orig")

    for value, g in grouped:

        total = len(g)
        cross_count = g["cross"].sum()

        ratio = cross_count / total

        if ratio > 0.5:

            # 修改 CT 值
            idx = g.index
            ct_df.loc[idx, feature] = 0

            # 紀錄
            report_rows.append({
                "feature": feature,
                "original_value": value,
                "count": total,
                "cross_count": cross_count,
                "cross_ratio": ratio,
                "original_CT_mean": g["ct"].mean()
            })

# ===== 新增 sum =====
ct_df["sum"] = ct_df[feature_cols].sum(axis=1)

# 儲存新的 CT
ct_df.to_csv(new_ct_csv, index=False)

# 儲存報告
report_df = pd.DataFrame(report_rows)
report_df.to_csv(report_csv, index=False)

print("Done.")