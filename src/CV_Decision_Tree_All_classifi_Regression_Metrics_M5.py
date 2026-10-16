"""Five-fold stratified Decision Tree evaluation for GlucoTrend classification."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Save a visual snapshot of the cross-validation scores and pooled predictions.
charts = os.path.join(OUTPUT, "charts")
os.makedirs(charts, exist_ok=True)
metric_names = ["Accuracy", "Balanced Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score (weighted)", "ROC-AUC"]
available_metrics = [(name, metrics[name]) for name in metric_names if name in metrics and np.isfinite(metrics[name])]
matrix = confusion_matrix(y, predicted, labels=classes)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
fig.patch.set_facecolor("#101c20")
for axis in axes:
    axis.set_facecolor("#101c20")
    axis.tick_params(colors="#dce9e5")
    for spine in axis.spines.values():
        spine.set_color("#385057")
bars = axes[0].barh([name for name, _ in available_metrics], [score * 100 for _, score in available_metrics], color="#75d5bc")
axes[0].invert_yaxis()
axes[0].set_xlim(0, 105)
axes[0].set_xlabel("Score (%)", color="#dce9e5")
axes[0].set_title("5-fold stratified cross-validation", color="#e7f2ef", loc="left", pad=14, weight="bold")
axes[0].grid(axis="x", color="#294047", alpha=0.8)
for bar, (_, score) in zip(bars, available_metrics):
    axes[0].text(min(score * 100 + 0.8, 100), bar.get_y() + bar.get_height() / 2, f"{score * 100:.2f}%", va="center", color="#dce9e5", fontsize=9)
image = axes[1].imshow(matrix, cmap="BuGn")
axes[1].set_title("Pooled confusion matrix", color="#e7f2ef", loc="left", pad=14, weight="bold")
axes[1].set_xticks(range(len(classes)), classes, rotation=20, ha="right", color="#dce9e5")
axes[1].set_yticks(range(len(classes)), classes, color="#dce9e5")
axes[1].set_xlabel("Predicted", color="#dce9e5")
axes[1].set_ylabel("Actual", color="#dce9e5")
threshold = matrix.max() / 2 if matrix.size else 0
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        axes[1].text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                     color="#ffffff" if matrix[row, col] > threshold else "#102126", fontsize=10, weight="bold")
fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle("M5 · Model evaluation", x=0.04, ha="left", color="#76d7bd", fontsize=11, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(charts, "cross_validation_snapshot.png"), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print("Five-fold stratified Decision Tree evaluation completed.")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}" if np.isfinite(value) else f"{name}: unavailable")
print(f"Outputs: {OUTPUT}")
