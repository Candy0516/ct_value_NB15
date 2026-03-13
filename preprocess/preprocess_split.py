# -*- coding: utf-8 -*-
"""
UNSW-NB15 preprocessing (train / val / test ready)
- Label encoding for nominal features
- Min-Max normalization (per feature)
- Preserve original dataframe column order
- Binary label as last column
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# =====================================================
#  Nominal Label Encoding
# =====================================================
def label_encode_nominal(
    dataframe,
    nominal_columns,
    encoders=None,
    fit=True,
    fill_value="unknown"
):
    if encoders is None:
        encoders = {}

    encoded_cols = []

    for col in nominal_columns:
        if col not in dataframe.columns:
            continue

        # 補缺失值 + 轉字串
        values = dataframe[col].fillna(fill_value).astype(str)

        if fit:
            le = LabelEncoder()

            # 確保 unknown 一定存在於 classes_
            values_with_unknown = values.copy()
            if fill_value not in values_with_unknown.values:
                values_with_unknown = pd.concat(
                    [values_with_unknown, pd.Series([fill_value])],
                    ignore_index=True
                )

            le.fit(values_with_unknown)
            encoders[col] = le

            encoded = le.transform(values)

        else:
            le = encoders[col]

            # 👉 把 unseen labels 映射成 unknown
            values = values.where(
                values.isin(le.classes_),
                other=fill_value
            )

            encoded = le.transform(values)

        encoded_cols.append(encoded.reshape(-1, 1))

    if len(encoded_cols) == 0:
        nominal_x = np.empty((len(dataframe), 0), dtype=np.float32)
    else:
        nominal_x = np.hstack(encoded_cols).astype(np.float32)

    return nominal_x, encoders


# =====================================================
#  Main preprocessing function
# =====================================================
def preprocess_nb15_dataframe(
    dataframe,
    nominal_names,
    integer_names,
    float_names,
    binary_names,
    *,
    label_col="label",
    encoders=None,
    scaler=None,
    fit=True
):
    df = dataframe.copy()

    # -------------------------------------------------
    # Label encode nominal
    # -------------------------------------------------
    nominal_x, encoders = label_encode_nominal(
        df,
        nominal_names,
        encoders=encoders,
        fit=fit
    )

    for i, col in enumerate(nominal_names):
        df[col] = nominal_x[:, i].astype(np.float32)

    # -------------------------------------------------
    # Build feature order (MUST match X)
    # -------------------------------------------------
    feature_order = (
        list(integer_names) +
        list(nominal_names) +
        list(float_names) +
        list(binary_names)
    )

    # ensure numeric dtype
    for col in feature_order:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_order].values.astype(np.float32)

    # -------------------------------------------------
    # Remove NaN rows
    # -------------------------------------------------
    nan_rows = np.unique(np.where(np.isnan(X))[0])
    if len(nan_rows) > 0:
        df = df.drop(df.index[nan_rows]).reset_index(drop=True)
        X = np.delete(X, nan_rows, axis=0)

    # -------------------------------------------------
    # Min-Max normalization (per feature)
    # -------------------------------------------------
    if fit:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # -------------------------------------------------
    # Write normalized values back
    # -------------------------------------------------
    df_norm = pd.DataFrame(X_scaled, columns=feature_order)
    for col in feature_order:
        df[col] = df_norm[col].values.astype(np.float32)

    # -------------------------------------------------
    # Output dataframe (original order + label)
    # -------------------------------------------------
    feature_cols_in_order = [c for c in df.columns if c in feature_order]

    df_out = df.loc[:, feature_cols_in_order].copy()
    df_out[label_col] = df[label_col].values.astype(np.int16)

    return df_out, encoders, scaler


# =====================================================
#  Load feature definition
# =====================================================
print("Reading feature info...")
feat_df = pd.read_csv(
    "NUSW-NB15_features.csv",
    encoding="ISO-8859-1",
    header=None
)

features = feat_df.values[1:-2]
feature_names = np.array([str(f).strip().lower() for f in features[:, 1]])
feature_types = np.array([t.lower() for t in features[:, 2]])

alias_map = {
    "dintpkt": "dinpkt",
    "sintpkt": "sinpkt",
    "smeansz": "smean",
    "dmeansz": "dmean",
    "ct_src_ ltm": "ct_src_ltm",
    "res_bdy_len": "response_body_len"
}

feature_names = np.array([alias_map.get(f, f) for f in feature_names])

nominal_names = feature_names[feature_types == "nominal"].tolist()
integer_names = feature_names[feature_types == "integer"].tolist()
float_names   = feature_names[feature_types == "float"].tolist()
binary_names  = feature_names[feature_types == "binary"].tolist()


# =====================================================
#  Load datasets
# =====================================================
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")
#  合併官方 train + test
# =====================================================
full_df = pd.concat([train_df, test_df], ignore_index=True)

print("Full dataset shape:", full_df.shape)
print("Label distribution:")
print(full_df['label'].value_counts())

# =====================================================
#  include derived feature (e.g., rate)
# =====================================================
if "rate" in full_df.columns and "rate" not in float_names:
    float_names.append("rate")

# keep only existing columns
nominal_names = [c for c in nominal_names if c in full_df.columns]
integer_names = [c for c in integer_names if c in full_df.columns]
float_names   = [c for c in float_names   if c in full_df.columns]
binary_names  = [c for c in binary_names  if c in full_df.columns]

# =====================================================
#  切分：60% train / 20% val / 20% test
# =====================================================

# Step 1: 先切 60% TRAIN + 40% TEMP
train_df_split, temp_df = train_test_split(
    full_df,
    test_size=0.4,
    random_state=42,
    stratify=full_df['label']
)

# Step 2: 再把 TEMP 均分成 VAL / TEST
val_df, test_df_split = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['label']
)

print("\nSplit sizes:")
print("TRAIN:", train_df_split.shape)
print("VAL  :", val_df.shape)
print("TEST :", test_df_split.shape)

# =====================================================
#  Preprocess TRAIN (fit)
# =====================================================
print("Processing TRAIN...")
train_proc, encoders, scaler = preprocess_nb15_dataframe(
    train_df_split,
    nominal_names,
    integer_names,
    float_names,
    binary_names,
    fit=True
)

train_proc.to_csv(
    "UNSW_NB15_train_split_all.csv",
    index=False
)

print("Processing VAL...")
val_proc, _, _ = preprocess_nb15_dataframe(
    val_df,
    nominal_names,
    integer_names,
    float_names,
    binary_names,
    encoders=encoders,
    scaler=scaler,
    fit=False
)

val_proc.to_csv(
    "UNSW_NB15_val_split_all.csv",
    index=False
)

# =====================================================
#  Preprocess TEST (transform only)
# =====================================================
print("Processing TEST...")
test_proc, _, _ = preprocess_nb15_dataframe(
    test_df_split,
    nominal_names,
    integer_names,
    float_names,
    binary_names,
    encoders=encoders,
    scaler=scaler,
    fit=False
)

test_proc.to_csv(
    "UNSW_NB15_test_split_all.csv",
    index=False
)

print("Done.")
print("TRAIN:", train_proc.shape)
print("VAL  :", val_proc.shape)
print("TEST :", test_proc.shape)
