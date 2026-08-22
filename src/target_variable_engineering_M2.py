# ============================================================
# GLUCOTREND DATASET - TARGET VARIABLE ENGINEERING
#
# Unlike placement_predict_50K_Raw.csv (which already contained
# a ready-made target column "PlacementStatus"), the GlucoBench
# raw CGM data has NO target column - it only has past features
# (glucose_lag_1, glucose_lag_3, glucose_lag_6, glucose_roll_mean_1h).
#
# This script BUILDS every candidate target variable so you can
# choose which framing to model:
#   1. REGRESSION   -> glucose_lead_1 / lead_3 / lead_6 (future glucose)
#   2. BINARY        -> target_binary        (In Range vs Out of Range)
#   3. 3-CLASS (ADA)  -> target_ada3          (Hypo / Target / Hyper)
#   4. 5-CLASS        -> target_5class        (ATTD/ADA clinical tiers)
#   5. TREND-BASED    -> target_trend         (rate-of-change direction)
#
# Original raw dataset is NOT modified.
# All targets are computed PER user_id (grouped) so no future
# information ever leaks across different patients.
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "GlucoBench_benchmark_dataset_RAW.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
REPORT_FOLDER = os.path.join(BASE_DIR, "outputs", "Target_Variable_Analysis_M2")
os.makedirs(REPORT_FOLDER, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

print("=" * 70)
print("GLUCOTREND - TARGET VARIABLE ENGINEERING")
print("=" * 70)
print("\nRaw Dataset Shape:", df.shape)

# Sort by patient and time - REQUIRED before any shift()/diff() below
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ============================================================
# TARGET 1: REGRESSION / FORECASTING TARGETS
# ============================================================
# Mirrors the existing glucose_lag_1 / lag_3 / lag_6 (look-back)
# features with the equivalent look-ahead ("lead") targets.
# glucose_lead_1 = the very next CGM reading for that same patient.
# This is the natural target for a forecasting benchmark like
# GlucoBench (predict the next glucose value from history).

for horizon in [1, 3, 6]:
    df[f"glucose_lead_{horizon}"] = df.groupby("user_id")["glucose"].shift(-horizon)

# ============================================================
# TARGET 2: BINARY CLASSIFICATION
# ============================================================
# In-Range (70-180 mg/dL, the ADA "time in range" band) vs Out-of-Range.
# NOTE: see the analysis printed below - true hypoglycemia (<70) does
# not occur in this dataset (values are floored at 70.0), so in
# practice this binary flag mainly captures HYPERGLYCEMIA risk.

df["target_binary"] = np.where(
    (df["glucose"] < 70) | (df["glucose"] > 180),
    "Out_of_Range",
    "In_Range"
)

# ============================================================
# TARGET 3: THREE-CLASS (ADA TIME-IN-RANGE STANDARD)
# ============================================================
# Hypoglycemia   : glucose < 70
# Target Range   : 70 <= glucose <= 180
# Hyperglycemia  : glucose > 180


def ada_3class(g):
    if g < 70:
        return "Hypoglycemia"
    elif g <= 180:
        return "Target Range"
    else:
        return "Hyperglycemia"


df["target_ada3"] = df["glucose"].apply(ada_3class)

# ============================================================
# TARGET 4: FIVE-CLASS (ATTD / ADA CLINICAL CGM CONSENSUS TIERS)
# ============================================================
# Level 2 Hypoglycemia : glucose < 54
# Level 1 Hypoglycemia : 54 <= glucose < 70
# Target Range         : 70 <= glucose <= 180
# Level 1 Hyperglycemia: 180 < glucose <= 250
# Level 2 Hyperglycemia: glucose > 250


def clinical_5class(g):
    if g < 54:
        return "L2_Hypoglycemia"
    elif g < 70:
        return "L1_Hypoglycemia"
    elif g <= 180:
        return "Target_Range"
    elif g <= 250:
        return "L1_Hyperglycemia"
    else:
        return "L2_Hyperglycemia"


df["target_5class"] = df["glucose"].apply(clinical_5class)

# ============================================================
# TARGET 5: TREND-BASED (RATE-OF-CHANGE DIRECTION)
# ============================================================
# Real CGM devices (Dexcom/Libre) show trend arrows using FIXED
# thresholds (e.g. Rising Rapidly >= +2 mg/dL/min). This synthetic
# dataset is far less volatile than a real sensor feed, so fixed
# clinical thresholds collapse ~99.97% of rows into "Stable".
# Instead we use DATA-DRIVEN quantile thresholds (computed below)
# on the per-patient rate of change, which gives a usable, roughly
# balanced 5-level trend label.

df["prev_glucose"] = df.groupby("user_id")["glucose"].shift(1)
df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
delta_minutes = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60
rate_of_change = (df["glucose"] - df["prev_glucose"]) / delta_minutes
df["glucose_roc_per_min"] = rate_of_change

q20, q40, q60, q80 = rate_of_change.quantile([0.20, 0.40, 0.60, 0.80])


def trend_label(r):
    if pd.isna(r):
        return np.nan
    elif r <= q20:
        return "Falling Rapidly"
    elif r <= q40:
        return "Falling"
    elif r <= q60:
        return "Stable"
    elif r <= q80:
        return "Rising"
    else:
        return "Rising Rapidly"


df["target_trend"] = rate_of_change.apply(trend_label)

# Drop helper columns used only for calculation
df.drop(columns=["prev_glucose", "prev_timestamp"], inplace=True)

# ============================================================
# ANALYSIS REPORT (printed + saved) - class balance per framing
# ============================================================
report_lines = []
report_lines.append("=" * 70)
report_lines.append("TARGET VARIABLE CLASS BALANCE REPORT")
report_lines.append("=" * 70)

for col in ["target_binary", "target_ada3", "target_5class", "target_trend"]:
    counts = df[col].value_counts(dropna=False)
    pct = (df[col].value_counts(normalize=True, dropna=False) * 100).round(2)
    report_lines.append(f"\n--- {col} ---")
    for label in counts.index:
        report_lines.append(f"  {label:<20} count={counts[label]:>6}   pct={pct[label]}%")

for horizon in [1, 3, 6]:
    col = f"glucose_lead_{horizon}"
    report_lines.append(f"\n--- {col} (regression target) ---")
    report_lines.append(f"  non-null rows: {df[col].notnull().sum()} / {len(df)}")
    report_lines.append(f"  mean={df[col].mean():.2f}  std={df[col].std():.2f}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(os.path.join(REPORT_FOLDER, "target_class_balance_report.txt"), "w") as f:
    f.write(report_text)

# ============================================================
# SAVE TARGET-ENGINEERED DATASET
# ============================================================
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("TARGET VARIABLE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Rows in:", len(pd.read_csv(INPUT_FILE)), " Rows out:", len(df))
print("Saved file:", OUTPUT_FILE)
print("Class balance report:", os.path.join(REPORT_FOLDER, "target_class_balance_report.txt"))
print("\nDownstream scripts (clean_*, final_preprocess_M2.py) should now")
print("read 'target_engineered_M2.csv' instead of the raw CSV, and set")
print("target_column to whichever of the 5 candidates above you choose.")
