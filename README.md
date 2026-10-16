# GlucoTrend

CGM (continuous glucose monitoring) ML project on the GlucoBench benchmark
dataset (10 patients, 15,731 readings). Structure mirrors placement_prediction.

## Quick start
```
pip install pandas numpy scikit-learn matplotlib seaborn flask
python main.py      # runs the full src/ pipeline, regenerates dataset/ + outputs/
python app.py        # launches the Flask app at http://127.0.0.1:5000
```

## Web app
The Flask app is fully wired to real data — nothing is a static placeholder:
- **/dataset** — live stats + row preview computed from the raw CSV
- **/preprocessing** — pipeline steps with live done/not-run status per script
- **/visualization** — chart gallery plus the complete, searchable output list
- **/models** — supervised, regression, clustering, anomaly, and dimensionality methods with saved evaluation metrics
- **/prediction** — a real form; submits to the trained Decision Tree pipeline and
  returns a live In_Range / Out_of_Range prediction with class probabilities
- **/dashboard** — aggregated headline numbers (best model, class balance, etc.)
- **/reports** — lists and serves real downloadable report files

The Models page compares classification, regression, and unsupervised metrics and links
to a downloadable output artifact for each method. The original pasted material is kept in
`reference/Pasted_Codes_Reference.txt` for completeness. It includes placement-prediction
scripts and generic Python examples whose Windows paths and target assumptions do not
match GlucoTrend; the runnable GlucoTrend pipeline is the project-root `main.py` and
the adapted scripts under `src/`.

## Target variables (see src/target_variable_engineering_M2.py)
| Column | Framing | Notes |
|---|---|---|
| glucose_lead_1 / 3 / 6 | Regression / forecasting | Recommended - matches existing lag features |
| target_binary | Binary | In_Range vs Out_of_Range, ~97/3 imbalance. Used by the live predictor. |
| target_ada3 | 3-class (ADA) | Hypoglycemia class is EMPTY in this data (floor-clipped at 70) |
| target_5class | 5-class (ATTD/ADA tiers) | Only 3 of 5 classes ever occur - same clipping issue |
| target_trend | Trend / rate-of-change | Quantile-calibrated (fixed clinical thresholds give ~100% "Stable") |

Full class-balance numbers: outputs/Target_Variable_Analysis_M2/target_class_balance_report.txt

## Folder layout
```
dataset/            raw + all intermediate/processed CSVs
notebooks/           Load_dataset_identify_missing_values.ipynb
outputs/             EDA images, target class-balance report, Module 3/4 model outputs
models/              (empty - trained model files go here)
general_programs/    (empty - mirrors placement_prediction)
src/                  all pipeline scripts (M1 = explore, M2 = engineer/clean, M3 = classifiers, M4 = clustering)
static/, templates/, app.py, main.py   Flask app (premium theme, real data throughout)
```

## Module 3 - classification (trained on target_binary)
- `Decision_Tree_Classifier_M3.py` — powers the live /prediction form
- `Random_Forest_Tree_M3.py`
- `AdaBoost_Classifier_M3.py`

Each drops `glucose` from the feature set (the target is derived directly from it) plus the
unused target candidates — see the leakage note at the top of each file. Change `TARGET_COLUMN`
inside any of them to retrain on `target_ada3`, `target_5class`, or `target_trend` instead.

## Module 4 - clustering
- `K_Means_Elbow_Silhouetee_M4.py` — K-Means & K-Means++ on glucose / heart_rate /
  stress_level / hbA1c, with Elbow, Silhouette, 3-D, and PCA visualizations.
- `Anomaly_Detection_Isolat_Forest_M4.py`, `PCA_M4.py`, `t-SNE_M4.py`,
  `DBSCAN_Raw_datasetM4.py`, and `Hierarchical_cluste_Agglomerative_divisive_M4.py`
  — GlucoBench-compatible anomaly detection, dimensionality reduction, and both
  agglomerative and divisive clustering.

## Module 5 - cross-validation
- `CV_Decision_Tree_All_classifi_Regression_Metrics_M5.py` — five-fold stratified
  Decision Tree evaluation for the GlucoTrend `target_binary` task. Outputs include
  weighted classification metrics, balanced accuracy, ROC-AUC, predictions, and a
  confusion matrix. Regression-only metrics from the placement example are not used
  for this classification target.

## Module 2 - linear models
- `Linear_Regression_M2.py` and `Lasso_Regression_M2.py` — next-reading glucose forecasts,
  compared with MAE, RMSE, R², and MAPE.
- `Logistic_Regression_M2.py` — in-range / out-of-range classification.
- `Multinomial_Regression_M2.py` — five-level glucose trend classification.
Each script saves predictions, metrics, a serialized pipeline, and diagnostic plots under
its matching `outputs/*_M2_Outputs/` folder.

`python main.py` runs the complete M1–M5 workflow, including the Module 2 linear models. The Module 4 sample-based analyses
write their artifacts under `outputs/GlucoTrend_Module_4_Outputs/`.
