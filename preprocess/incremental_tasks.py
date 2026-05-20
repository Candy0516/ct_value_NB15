import os
import pandas as pd

# ===============================
# 路徑設定
# ===============================
base_dir = "D:/candy/NB15"   # 改成你的資料夾

train_path = os.path.join(base_dir, "data", "1_preprocess", "data_split_val_attackcat", "train.csv")
test_path  = os.path.join(base_dir, "data", "1_preprocess", "data_split_val_attackcat", "test.csv")
val_path   = os.path.join(base_dir, "data", "1_preprocess", "data_split_val_attackcat", "val.csv")

output_dir = os.path.join(base_dir, "compare", "incremental_tasks")
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 讀取資料
# ===============================
train_df = pd.read_csv(train_path, low_memory=False)
test_df  = pd.read_csv(test_path, low_memory=False)
val_df   = pd.read_csv(val_path, low_memory=False)

# ===============================
# 檢查欄位
# ===============================
for name, df in [("train", train_df), ("test", test_df), ("val", val_df)]:
    if "attack_cat" not in df.columns:
        raise ValueError(f"{name}.csv 找不到 attack_cat 欄位")

# ===============================
# Task 類別設定
# ===============================
task_classes = {
    "task1": ["Normal", "Fuzzers", "Backdoor"],
    "task2": ["Normal", "Fuzzers", "Backdoor", "DoS", "Exploits", "Analysis"],
    "task3": ["Normal", "Fuzzers", "Backdoor", "DoS", "Exploits", "Analysis",
              "Generic", "Reconnaissance", "Shellcode"]
}

datasets = {
    "train": train_df,
    "test": test_df,
    "val": val_df
}

# ===============================
# 依 task 分別篩選並輸出
# ===============================
task3_all = []

for task_name, classes in task_classes.items():
    print(f"\n========== {task_name} ==========")

    for split_name, df in datasets.items():
        task_df = df[df["attack_cat"].isin(classes)].copy()

        save_path = os.path.join(output_dir, f"{split_name}_{task_name}.csv")
        task_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"{split_name}_{task_name}: {len(task_df)} rows -> {save_path}")

        # 刪除 attack_cat版本
        task_df_drop = task_df.drop(columns=["attack_cat"])

        save_path = os.path.join(output_dir, f"{split_name}_{task_name}_nocat.csv")
        task_df_drop.to_csv(save_path, index=False, encoding="utf-8-sig")

        if task_name == "task3":
            task3_all.append(task_df)

# ===============================
# 分別輸出 task3 每個 attack_cat 數量
# Train / Test / Val 分開統計
# ===============================
print("\n========== Task3 attack_cat counts by split ==========")

for split_name, df in datasets.items():
    task3_df = df[df["attack_cat"].isin(task_classes["task3"])].copy()

    counts = task3_df["attack_cat"].value_counts().sort_index()

    print(f"\n----- {split_name.upper()} Task3 attack_cat counts -----")
    print(counts)

    counts_path = os.path.join(
        output_dir,
        f"{split_name}_task3_attack_cat_counts.csv"
    )

    counts.to_csv(counts_path, header=["count"], encoding="utf-8-sig")

    print(f"Saved to: {counts_path}")