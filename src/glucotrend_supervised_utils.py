"""Shared data preparation and artifact helpers for linear model examples."""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "dataset", "target_engineered_M2.csv")
OUTPUTS = os.path.join(BASE_DIR, "outputs")
TARGET_COLUMNS = {
    "glucose_lead_1", "glucose_lead_3", "glucose_lead_6",
    "target_binary", "target_ada3", "target_5class", "target_trend",
}
NON_FEATURE_COLUMNS = {"user_id", "timestamp", "notes", *TARGET_COLUMNS}


def load_split(target, *, exclude_current_glucose=False, split_by_patient=True):
    if not os.path.isfile(DATASET):
        raise FileNotFoundError("Run target_variable_engineering_M2.py first.")
    data = pd.read_csv(DATASET).dropna(subset=[target]).copy()
    excluded = set(NON_FEATURE_COLUMNS)
    if exclude_current_glucose:
        # Prevent the present reading or its derived rollup/rate from revealing
        # a same-reading class label. Historical lag features remain available.
        excluded.update({"glucose", "glucose_roll_mean_1h", "glucose_roc_per_min"})
    feature_columns = [column for column in data.columns if column not in excluded]
    X = data[feature_columns].copy()
    y = data[target].copy()
    groups = data["user_id"] if "user_id" in data else pd.Series(np.arange(len(data)))

    if split_by_patient:
        splitter = GroupShuffleSplit(n_splits=120, test_size=0.2, random_state=42)
        all_classes = set(pd.Series(y).astype(str))
        best_split = None
        best_score = None
        for train_candidate, test_candidate in splitter.split(X, y, groups):
            train_classes = set(pd.Series(y.iloc[train_candidate]).astype(str))
            test_classes = set(pd.Series(y.iloc[test_candidate]).astype(str))
            test_counts = pd.Series(y.iloc[test_candidate]).astype(str).value_counts()
            score = (len(test_classes), min(test_counts) if len(test_counts) else 0,
                     len(train_classes & all_classes))
            if train_classes >= all_classes and (best_score is None or score > best_score):
                best_split, best_score = (train_candidate, test_candidate), score
                if len(test_classes) == len(all_classes) and min(test_counts) >= 25:
                    break
        if best_split is None:
            best_split = next(splitter.split(X, y, groups))
        train_idx, test_idx = best_split
    else:
        all_idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, stratify=y)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], preprocessor


def output_dir(model_key):
    path = os.path.join(OUTPUTS, f"{model_key}_M2_Outputs")
    for child in ("metrics", "predictions", "charts", "model"):
        os.makedirs(os.path.join(path, child), exist_ok=True)
    return path


def save_metrics(metrics, folder, filename):
    pd.DataFrame([{"Metric": name, "Value": value} for name, value in metrics.items()]).to_csv(
        os.path.join(folder, "metrics", filename), index=False)


def save_model(model, folder, filename):
    with open(os.path.join(folder, "model", filename), "wb") as handle:
        pickle.dump(model, handle)


def save_regression_artifacts(model_name, model_key, y_test, predicted, metrics, estimator):
    folder = output_dir(model_key)
    save_metrics(metrics, folder, f"{model_key.lower()}_metrics.csv")
    pd.DataFrame({"actual_glucose": y_test.to_numpy(), "predicted_glucose": predicted,
                  "residual": y_test.to_numpy() - predicted}).to_csv(
        os.path.join(folder, "predictions", "test_predictions.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, predicted, s=9, alpha=.35, color="#278e7c")
    lower = float(min(y_test.min(), np.min(predicted)))
    upper = float(max(y_test.max(), np.max(predicted)))
    axes[0].plot([lower, upper], [lower, upper], "--", color="#c47b41", linewidth=1.5)
    axes[0].set(title=f"{model_name}: actual vs predicted", xlabel="Actual next glucose (mg/dL)", ylabel="Predicted (mg/dL)")
    axes[1].scatter(predicted, y_test.to_numpy() - predicted, s=9, alpha=.35, color="#526eb2")
    axes[1].axhline(0, linestyle="--", color="#c47b41", linewidth=1.2)
    axes[1].set(title="Residuals", xlabel="Predicted glucose (mg/dL)", ylabel="Actual − predicted (mg/dL)")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "charts", "regression_diagnostics.png"), dpi=180)
    plt.close(fig)
    save_model(estimator, folder, f"{model_key.lower()}_model.pkl")
    return folder


def save_classification_artifacts(model_name, model_key, y_test, predicted, metrics, estimator):
    from sklearn.metrics import classification_report, confusion_matrix

    folder = output_dir(model_key)
    save_metrics(metrics, folder, f"{model_key.lower()}_metrics.csv")
    classes = sorted(set(pd.Series(y_test).astype(str)) | set(pd.Series(predicted).astype(str)))
    pd.DataFrame({"actual": pd.Series(y_test).astype(str).to_numpy(),
                  "predicted": pd.Series(predicted).astype(str).to_numpy()}).to_csv(
        os.path.join(folder, "predictions", "test_predictions.csv"), index=False)
    report = classification_report(y_test, predicted, labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(folder, "metrics", "classification_report.csv"))
    matrix = confusion_matrix(y_test, predicted, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        os.path.join(folder, "metrics", "confusion_matrix.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    score_names = ["Accuracy", "Precision", "Recall", "F1"]
    score_values = [metrics[name] for name in score_names]
    axes[0].bar(score_names, score_values, color=["#46a78e", "#67b79d", "#5087a2", "#d4a15e"])
    axes[0].set(ylim=(0, 1), ylabel="Score", title=f"{model_name}: holdout scores")
    axes[0].tick_params(axis="x", rotation=20)
    image = axes[1].imshow(matrix, cmap="GnBu")
    axes[1].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual",
                xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
                xticklabels=classes, yticklabels=classes)
    axes[1].tick_params(axis="x", rotation=30)
    for (row, col), value in np.ndenumerate(matrix):
        axes[1].text(col, row, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "charts", "classification_metrics.png"), dpi=180)
    plt.close(fig)
    save_model(estimator, folder, f"{model_key.lower()}_model.pkl")
    return folder
