# GlucoTrend

CGM (continuous glucose monitoring) ML project on the GlucoBench benchmark
dataset (10 patients, 15,731 readings). Structure mirrors placement_prediction.

## Quick start
```
python main.py      # runs the full src/ pipeline, regenerates dataset/ + outputs/
python app.py        # launches the Flask app (routes mirror placement_prediction's)
```

## Target variables (see src/target_variable_engineering_M2.py)
| Column | Framing | Notes |
|---|---|---|
| glucose_lead_1 / 3 / 6 | Regression / forecasting | Recommended - matches existing lag features |
| target_binary | Binary | In_Range vs Out_of_Range, ~97/3 imbalance |
| target_ada3 | 3-class (ADA) | Hypoglycemia class is EMPTY in this data (floor-clipped at 70) |
| target_5class | 5-class (ATTD/ADA tiers) | Only 3 of 5 classes ever occur - same clipping issue |
| target_trend | Trend / rate-of-change | Quantile-calibrated (fixed clinical thresholds give ~100% "Stable") |

Full class-balance numbers: outputs/Target_Variable_Analysis_M2/target_class_balance_report.txt

## Folder layout
```
dataset/            raw + all intermediate/processed CSVs
notebooks/           Load_dataset_identify_missing_values.ipynb
outputs/             EDA images, target class-balance report
models/              (empty - trained model files go here)
general_programs/    (empty - mirrors placement_prediction)
src/                  all pipeline scripts (M1 = explore, M2 = engineer/clean)
static/, templates/, app.py, main.py   Flask app skeleton
```
