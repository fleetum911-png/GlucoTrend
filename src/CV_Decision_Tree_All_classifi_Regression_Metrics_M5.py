"""Five-fold stratified Decision Tree evaluation for GlucoTrend classification."""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
OUTPUT = os.path.join(BASE_DIR, "outputs", "CV_Decision_Tree_M5_Outputs")
TARGET = "target_binary"

if not os.path.exists(DATASET):
    raise FileNotFoundError(f"Run target_variable_engineering_M2.py first: {DATASET}")
data = pd.read_csv(DATASET).dropna(subset=[TARGET]).copy()
leakage = ["glucose", "glucose_roc_per_min", "target_ada3", "target_5class", "target_trend",
           "glucose_lead_1", "glucose_lead_3", "glucose_lead_6", "user_id", "timestamp", "notes"]
X = data.drop(columns=[TARGET] + leakage, errors="ignore")
y = data[TARGET].astype(str)
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
transform = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
], remainder="drop")
model = Pipeline([("features", transform), ("classifier", DecisionTreeClassifier(criterion="entropy", random_state=42))])
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predicted = cross_val_predict(model, X, y, cv=folds, method="predict", n_jobs=1)
probabilities = cross_val_predict(model, X, y, cv=folds, method="predict_proba", n_jobs=1)
classes = np.unique(y)
metrics = {
    "Accuracy": accuracy_score(y, predicted),
    "Balanced Accuracy": balanced_accuracy_score(y, predicted),
    "Precision (weighted)": precision_score(y, predicted, average="weighted", zero_division=0),
    "Recall (weighted)": recall_score(y, predicted, average="weighted", zero_division=0),
    "F1 Score (weighted)": f1_score(y, predicted, average="weighted", zero_division=0),
}
try:
    if len(classes) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y, probabilities[:, 1])
    else:
        metrics["ROC-AUC (weighted OVR)"] = roc_auc_score(y, probabilities, multi_class="ovr", average="weighted", labels=classes)
except ValueError:
    metrics["ROC-AUC (weighted OVR)"] = float("nan")
os.makedirs(OUTPUT, exist_ok=True)
pd.DataFrame([{"Metric": key, "Score": value, "Percentage": value * 100 if np.isfinite(value) else np.nan}
              for key, value in metrics.items()]).to_csv(os.path.join(OUTPUT, "cross_validation_metrics.csv"), index=False)
pd.DataFrame({"actual": y, "predicted": predicted}).to_csv(os.path.join(OUTPUT, "cross_validated_predictions.csv"), index=False)
pd.DataFrame(confusion_matrix(y, predicted, labels=classes), index=classes, columns=classes).to_csv(
    os.path.join(OUTPUT, "confusion_matrix.csv"))
with open(os.path.join(OUTPUT, "classification_report.txt"), "w") as report:
    report.write(classification_report(y, predicted, zero_division=0))
print("Five-fold stratified Decision Tree evaluation completed.")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}" if np.isfinite(value) else f"{name}: unavailable")
print(f"Outputs: {OUTPUT}")
