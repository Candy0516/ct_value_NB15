import os
import pandas as pd


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def replace_ct_with_zero_when_raw_dash_and_recalc_sum(raw_csv, ct_csv, save_csv):
    # 讀取原始資料與 CT 資料
    df_raw = pd.read_csv(raw_csv, low_memory=False)
    df_ct = pd.read_csv(ct_csv, low_memory=False)

    # 檢查列數
    if len(df_raw) != len(df_ct):
        raise ValueError(f"列數不一致：raw={len(df_raw)}, ct={len(df_ct)}")

    # 複製 CT 資料
    df_out = df_ct.copy()

    # 只處理共同欄位
    common_cols = [col for col in df_ct.columns if col in df_raw.columns]
    print(f"共同欄位數: {len(common_cols)}")

    # 若原始值為 '-'，對應 CT 值改為 0
    for col in common_cols:
        if col == "sum":
            continue
        dash_mask = df_raw[col].astype(str).str.strip() == '-'
        df_out.loc[dash_mask, col] = 0

    # 重新計算 sum 欄位
    # sum = 所有特徵欄位總和，不包含原本的 sum 欄位
    feature_cols = [col for col in df_out.columns if col != "sum"]

    # 若你有 label / type 這類非特徵欄位，也可以一起排除
    exclude_cols = {"label", "type", "y_true", "y_pred", "correct", "correct_adjusted"}
    feature_cols = [col for col in feature_cols if col not in exclude_cols]

    # 轉成數值後加總，非數值轉成 NaN 再補 0
    df_out[feature_cols] = df_out[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_out["sum"] = df_out[feature_cols].sum(axis=1)

    # 存檔
    df_out.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {save_csv}")


if __name__ == "__main__":
    raw_dir = r"D:\candy\TONIOT\TON_IOT_v2\Data_val\0_DataSet"
    ct_dir = r"D:\candy\TONIOT\data"
    save_dir = r"D:\candy\TONIOT\data\8_dash_ct_0"

    create_folder(save_dir)

    file_pairs = [
        {
            "name": "Trn",
            "raw": os.path.join(raw_dir, "Train_Test_Network_Trn_Big.csv"),
            "ct": os.path.join(ct_dir, "4_sum", "ct-value", "train.csv"),
            "save": os.path.join(save_dir, "train.csv"),
        },
        {
            "name": "Tst",
            "raw": os.path.join(raw_dir, "Train_Test_Network_Tst_Big.csv"),
            "ct": os.path.join(ct_dir, "7_sum_test", "benign_test.csv"),
            "save": os.path.join(save_dir, "test.csv"),
        },
        {
            "name": "Val",
            "raw": os.path.join(raw_dir, "Train_Test_Network_Val_Big.csv"),
            "ct": os.path.join(ct_dir, "7_sum_val", "benign_val.csv"),
            "save": os.path.join(save_dir, "val.csv"),
        },
    ]

    for item in file_pairs:
        print(f"Processing {item['name']} ...")
        replace_ct_with_zero_when_raw_dash_and_recalc_sum(
            raw_csv=item["raw"],
            ct_csv=item["ct"],
            save_csv=item["save"]
        )

    print("All done.")