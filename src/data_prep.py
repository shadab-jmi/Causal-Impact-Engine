# Data loading, treatment flag, feature encoding, and balance (SMD) checks

from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

NUMERIC_FEATURES = ["recency", "history"]
BINARY_FEATURES = ["mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel", "history_segment"]

OUTCOMES = ["visit", "conversion", "spend"]


# Load the raw Hillstrom dataset from data/raw/
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


# Add a binary treatment flag: any email = 1, 'No E-Mail' = 0
def add_treatment_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    return df


# Map 'history_segment' (e.g. '2) $100 - $200') to its leading rank number
def _history_segment_to_ordinal(series: pd.Series) -> pd.Series:
    return series.str.extract(r"^(\d+)").astype(int).iloc[:, 0]

# Return a purely-numeric feature matrix for models and balance checks

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        out[col] = df[col].astype(float)

    out["history_segment_ord"] = _history_segment_to_ordinal(df["history_segment"]).astype(float)

    for col in ["zip_code", "channel"]:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        out = pd.concat([out, dummies], axis=1)

    return out


# Standardized mean difference (SMD) for one feature between two groups
def standardized_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    mean_diff = treated.mean() - control.mean()
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return mean_diff / pooled_sd


# Compute |SMD| for every encoded covariate, flagging |SMD| < 0.1 as balanced
def balance_table(df: pd.DataFrame, treatment_col: str = "treatment") -> pd.DataFrame:
    encoded = encode_features(df)
    encoded[treatment_col] = df[treatment_col].values

    treated = encoded[encoded[treatment_col] == 1]
    control = encoded[encoded[treatment_col] == 0]

    rows = []
    for col in encoded.columns:
        if col == treatment_col:
            continue
        smd = standardized_mean_difference(treated[col], control[col])
        rows.append({"feature": col, "smd": smd, "abs_smd": abs(smd)})

    table = pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)
    table["balanced"] = table["abs_smd"] < 0.1
    return table
