"""
GlucoTrend - app.py

Same 10 routes as the placement_prediction reference app, but wired
to real data instead of static placeholders:
  - /dataset       reads the raw CSV and computes real stats
  - /visualization builds an image gallery from outputs/ on the fly
  - /models        compares all supervised and unsupervised methods from src/
  - /prediction    loads the trained Decision Tree pipeline and
                    actually runs inference on submitted form data
  - /dashboard     aggregates headline numbers from the above
  - /reports       lists real downloadable report files
"""

import os
import pickle
from uuid import uuid4

import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "glucotrend-local-session")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = os.path.join(DATASET_DIR, "uploads")
RAW_CSV = os.path.join(DATASET_DIR, "GlucoBench_benchmark_dataset_RAW.csv")
TARGET_ENGINEERED_CSV = os.path.join(DATASET_DIR, "target_engineered_M2.csv")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "Decision_Tree_Classifier_M3_Outputs", "model", "decision_tree_model.pkl")

# ================================================================
# LOAD THE TRAINED MODEL ONCE AT STARTUP
# ================================================================
_model = None
_model_error = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    except Exception as exc:  # pragma: no cover
        _model_error = str(exc)
else:
    _model_error = "Model file not found - run src/Decision_Tree_Classifier_M3.py first."

NUMERIC_FEATURES = [
    "cgm_quality_flag", "insulin_bolus", "insulin_basal", "carbs", "exercise_steps",
    "heart_rate", "skin_temp", "gsr", "stress_level", "alcohol", "hbA1c", "age",
    "weight", "carb_ratio", "insulin_sensitivity", "glucose_lag_1", "glucose_lag_3",
    "glucose_lag_6", "glucose_roll_mean_1h", "glucose_roc_per_min",
]
CATEGORICAL_OPTIONS = {
    "meal_type": ["breakfast", "lunch", "dinner", "snack", "none"],
    "exercise_intensity": ["none", "low", "medium", "high"],
    "sleep_stage": ["awake", "light", "deep", "REM"],
    "medication_other": ["none", "antihistamine", "beta_blocker", "steroid"],
    "device_id": ["Dexcom_G6", "Libre_2", "Medtronic_670G"],
    "sex": ["F", "M"],
    "timezone": ["America/New_York", "America/Los_Angeles", "Europe/London"],
    "region": ["USA", "UK"],
}
# Sensible pre-filled defaults (dataset medians / most common category)
FIELD_DEFAULTS = {
    "cgm_quality_flag": 1, "insulin_bolus": 0, "insulin_basal": 1.0, "carbs": 0,
    "exercise_steps": 0, "heart_rate": 85, "skin_temp": 36.5, "gsr": 1.5,
    "stress_level": 3, "alcohol": 0, "hbA1c": 7.5, "age": 35, "weight": 79,
    "carb_ratio": 10, "insulin_sensitivity": 48, "glucose_lag_1": 108,
    "glucose_lag_3": 108, "glucose_lag_6": 108, "glucose_roll_mean_1h": 108,
    "glucose_roc_per_min": 0.0, "meal_type": "none", "exercise_intensity": "none",
    "sleep_stage": "light", "medication_other": "none", "device_id": "Dexcom_G6",
    "sex": "F", "timezone": "America/New_York", "region": "USA",
}
FIELD_RANGES = {
    "insulin_bolus": (0, 11, 0.1), "insulin_basal": (0.8, 1.2, 0.01), "carbs": (0, 90, 1),
    "exercise_steps": (0, 5000, 10), "heart_rate": (60, 110, 1), "skin_temp": (35.5, 37.5, 0.1),
    "gsr": (0.5, 2.5, 0.01), "stress_level": (0, 5, 1), "alcohol": (0, 2, 1),
    "hbA1c": (6.0, 8.5, 0.1), "age": (18, 90, 1), "weight": (40, 130, 1),
    "carb_ratio": (5, 20, 1), "insulin_sensitivity": (30, 70, 1),
    "glucose_lag_1": (70, 250, 1), "glucose_lag_3": (70, 250, 1), "glucose_lag_6": (70, 250, 1),
    "glucose_roll_mean_1h": (70, 250, 1), "glucose_roc_per_min": (-3, 3, 0.01),
}
FIELD_LABELS = {
    "cgm_quality_flag": "CGM Signal Quality (0 = poor, 1 = good)",
    "insulin_bolus": "Insulin Bolus (units)", "insulin_basal": "Insulin Basal Rate (units/hr)",
    "carbs": "Carbs Consumed (g)", "meal_type": "Meal Type",
    "exercise_steps": "Steps (last interval)", "exercise_intensity": "Exercise Intensity",
    "heart_rate": "Heart Rate (bpm)", "skin_temp": "Skin Temperature (°C)",
    "gsr": "Galvanic Skin Response", "sleep_stage": "Sleep Stage",
    "stress_level": "Stress Level (0-5)", "alcohol": "Alcohol Units",
    "medication_other": "Other Medication", "hbA1c": "HbA1c (%)", "age": "Age (years)",
    "sex": "Sex", "weight": "Weight (kg)", "carb_ratio": "Carb Ratio (g per unit insulin)",
    "insulin_sensitivity": "Insulin Sensitivity Factor", "device_id": "CGM Device",
    "timezone": "Timezone", "region": "Region",
    "glucose_lag_1": "Glucose - 1 reading ago (mg/dL)",
    "glucose_lag_3": "Glucose - 3 readings ago (mg/dL)",
    "glucose_lag_6": "Glucose - 6 readings ago (mg/dL)",
    "glucose_roll_mean_1h": "Glucose - 1h Rolling Mean (mg/dL)",
    "glucose_roc_per_min": "Glucose Rate of Change (mg/dL/min)",
}
FORM_SECTIONS = [
    ("Recent Glucose History", ["glucose_lag_1", "glucose_lag_3", "glucose_lag_6",
                                 "glucose_roll_mean_1h", "glucose_roc_per_min", "cgm_quality_flag"]),
    ("Insulin & Meals", ["insulin_bolus", "insulin_basal", "carbs", "meal_type"]),
    ("Activity & Biosignals", ["exercise_steps", "exercise_intensity", "heart_rate",
                                "skin_temp", "gsr", "sleep_stage", "stress_level", "alcohol"]),
    ("Patient Profile", ["hbA1c", "age", "sex", "weight", "carb_ratio",
                          "insulin_sensitivity", "medication_other", "device_id", "timezone", "region"]),
]

MODEL_DEFS = [
    ("Decision Tree", "Decision_Tree_Classifier_M3_Outputs", "decision_tree_metrics.csv"),
    ("Random Forest", "Random_Forest_Tree_M3_Outputs", "random_forest_metrics.csv"),
    ("AdaBoost", "AdaBoost_Classifier_M3_Outputs", "adaboost_metrics.csv"),
]
ML_TOPIC_ORDER = {
    "M1 · Data lifecycle & exploration": 1,
    "EDA · Exploratory data analysis": 2,
    "M2 · Targets & feature preparation": 3,
    "M2 · Supervised linear models": 4,
    "M3 · Tree-based models": 5,
    "M4 · Unsupervised learning": 6,
    "M5 · Model evaluation": 7,
    "M6 · ML engineering & other artifacts": 8,
}


# ================================================================
# HELPERS
# ================================================================
def get_dataset_stats():
    if not os.path.exists(RAW_CSV):
        return None
    df = pd.read_csv(RAW_CSV)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "patients": df["user_id"].nunique() if "user_id" in df.columns else "-",
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "glucose_min": round(df["glucose"].min(), 1) if "glucose" in df.columns else "-",
        "glucose_max": round(df["glucose"].max(), 1) if "glucose" in df.columns else "-",
        "glucose_mean": round(df["glucose"].mean(), 1) if "glucose" in df.columns else "-",
        "preview": df.head(8).to_html(classes="preview-table", index=False, border=0),
    }


def get_target_balance():
    if not os.path.exists(TARGET_ENGINEERED_CSV):
        return None
    df = pd.read_csv(TARGET_ENGINEERED_CSV)
    if "target_binary" not in df.columns:
        return None
    counts = df["target_binary"].value_counts()
    total = counts.sum()
    return [{"label": label, "count": int(count), "pct": round(count / total * 100, 2)}
             for label, count in counts.items()]


def get_model_metrics():
    results = []
    for display_name, folder_name, metrics_file in MODEL_DEFS:
        metrics_path = os.path.join(OUTPUTS_DIR, folder_name, "metrics", metrics_file)
        row = {"name": display_name, "folder": folder_name, "available": False}
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            for _, metric_row in metrics_df.iterrows():
                row[metric_row["Metric"].lower().replace(" ", "_")] = round(metric_row["Percentage"], 2)
            row["available"] = True
        results.append(row)
    return results


def _read_first_row(relative_path):
    path = os.path.join(OUTPUTS_DIR, relative_path)
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if not frame.empty else None


def _classifier_scores(relative_path):
    row = _read_first_row(relative_path)
    scores = {"accuracy": None, "precision": None, "recall": None, "f1": None}
    if not row:
        return scores
    metric_file = os.path.join(OUTPUTS_DIR, relative_path)
    for _, metric in pd.read_csv(metric_file).iterrows():
        key = str(metric.get("Metric", "")).lower()
        value = metric.get("Percentage")
        if pd.isna(value) and not pd.isna(metric.get("Value")):
            value = float(metric["Value"]) * 100
        if pd.isna(value):
            continue
        if key == "accuracy":
            scores["accuracy"] = float(value)
        elif key.startswith("precision"):
            scores["precision"] = float(value)
        elif key.startswith("recall"):
            scores["recall"] = float(value)
        elif key.startswith("f1"):
            scores["f1"] = float(value)
    return scores


def _regression_scores(relative_path):
    row = _read_first_row(relative_path)
    if not row:
        return {"mae": None, "rmse": None, "r2": None, "mape": None}
    metric_path = os.path.join(OUTPUTS_DIR, relative_path)
    metrics = pd.read_csv(metric_path)
    values = {str(metric["Metric"]).lower(): float(metric["Value"])
              for _, metric in metrics.iterrows() if pd.notna(metric.get("Value"))}
    return {"mae": values.get("mae"), "rmse": values.get("rmse"),
            "r2": values.get("r2"), "mape": values.get("mape")}


def get_model_showcase():
    """Return every algorithm implemented in src with its saved evaluation evidence."""
    regression_definitions = [
        ("Linear Regression", "Supervised regression", "Linear baseline forecasting the next glucose reading from current and historical sensor features.", "Linear_Regression_M2_Outputs/metrics/linear_regression_metrics.csv", "Linear_Regression_M2_Outputs/charts/regression_diagnostics.png"),
        ("Lasso Regression", "Regularized regression", "L1-regularized glucose forecast that can shrink less useful feature coefficients toward zero.", "Lasso_Regression_M2_Outputs/metrics/lasso_regression_metrics.csv", "Lasso_Regression_M2_Outputs/charts/regression_diagnostics.png"),
    ]
    catalog = []
    for name, family, description, metrics_file, artifact in regression_definitions:
        scores = _regression_scores(metrics_file)
        catalog.append({"name": name, "family": family, "description": description, "module": "M2",
                        "accuracy": None, "precision": None, "recall": None, "f1": None,
                        "cluster_accuracy": None, "silhouette": None, "clusters": None, "inertia": None,
                        "anomaly_rate": None, "variance_2d": None, "sample_size": None, **scores,
                        "evaluation_data": "Participant holdout", "status": "Evaluated" if scores["mae"] is not None else "Not run",
                        "artifact": artifact})

    for name, description, output_key, metrics_filename, split_description in [
        ("Logistic Regression", "Logistic classifier for in-range versus out-of-range readings using historical features.", "Logistic_Regression_M2_Outputs", "logistic_regression_metrics.csv", "Stratified row holdout"),
        ("Multinomial Logistic Regression", "Multinomial logistic model for five glucose trend-direction classes.", "Multinomial_Regression_M2_Outputs", "multinomial_regression_metrics.csv", "Participant holdout"),
    ]:
        scores = _classifier_scores(f"{output_key}/metrics/{metrics_filename}")
        catalog.append({"name": name, "family": "Supervised classification", "description": description, "module": "M2",
                        **scores, "cluster_accuracy": None, "silhouette": None, "clusters": None, "inertia": None,
                        "anomaly_rate": None, "variance_2d": None, "sample_size": None,
                        "mae": None, "rmse": None, "r2": None, "mape": None,
                        "evaluation_data": split_description,
                        "status": "Evaluated" if any(value is not None for value in scores.values()) else "Not run",
                        "artifact": f"{output_key}/charts/classification_metrics.png"})

    definitions = [
        ("Decision Tree", "Supervised classification", "An interpretable tree that predicts the glucose range from the prepared features.", "M3", "Decision_Tree_Classifier_M3_Outputs/metrics/decision_tree_metrics.csv", "Decision_Tree_Classifier_M3_Outputs/decision_tree/decision_tree_diagram.png"),
        ("Random Forest", "Supervised classification", "An ensemble of randomized decision trees that combines their predictions.", "M3", "Random_Forest_Tree_M3_Outputs/metrics/random_forest_metrics.csv", "Random_Forest_Tree_M3_Outputs/random_forest_tree/random_forest_first_tree.png"),
        ("AdaBoost", "Supervised classification", "A sequence of shallow trees that focuses on observations missed by earlier rounds.", "M3", "AdaBoost_Classifier_M3_Outputs/metrics/adaboost_metrics.csv", "AdaBoost_Classifier_M3_Outputs/adaboost_base_tree/adaboost_base_tree.png"),
    ]
    for name, family, description, module, metrics_file, artifact in definitions:
        scores = _classifier_scores(metrics_file)
        catalog.append({"name": name, "family": family, "description": description, "module": module,
                        **scores, "cluster_accuracy": None, "silhouette": None, "clusters": None,
                        "inertia": None, "anomaly_rate": None, "variance_2d": None,
                        "mae": None, "rmse": None, "r2": None, "mape": None,
                        "evaluation_data": "5-fold CV" if "CV" in name else "Holdout",
                        "sample_size": None, "status": "Evaluated" if any(v is not None for v in scores.values()) else "Not run",
                        "artifact": artifact})

    def add_cluster(name, description, metrics_path, artifact, *, matching_accuracy=None, inertia=None):
        row = _read_first_row(metrics_path)
        catalog.append({"name": name, "family": "Clustering", "description": description, "module": "M4",
                        "accuracy": None, "precision": None, "recall": None, "f1": None,
                        "cluster_accuracy": matching_accuracy, "silhouette": float(row["silhouette"] if "silhouette" in row else row["Silhouette_Score"]) if row else None,
                        "clusters": int(row["clusters"] if "clusters" in row else row["Number_of_Clusters"]) if row else None,
                        "inertia": float(inertia if inertia is not None else row["Inertia"]) if row and (inertia is not None or "Inertia" in row) else None,
                        "anomaly_rate": None, "variance_2d": None, "sample_size": int(row.get("sample_records", 0)) or None if row else None,
                        "mae": None, "rmse": None, "r2": None, "mape": None, "evaluation_data": None,
                        "status": "Evaluated" if row else "Not run", "artifact": artifact})

    accuracy_path = os.path.join(OUTPUTS_DIR, "K_Means_K++Means_Elbow_Silhoute_M4_Outputs/Accuracy/glucose_range_cluster_matching_accuracy.csv")
    matching = {}
    if os.path.isfile(accuracy_path):
        accuracy_frame = pd.read_csv(accuracy_path)
        matching = {str(row["Method"]): float(row["Accuracy_Percentage"]) for _, row in accuracy_frame.iterrows()}
    add_cluster("K-Means", "Partitions readings into a selected number of groups by minimizing within-cluster distance.",
                "K_Means_K++Means_Elbow_Silhoute_M4_Outputs/KMeans/kmeans_metrics.csv",
                "K_Means_K++Means_Elbow_Silhoute_M4_Outputs/KMeans/kmeans_PCA_clustering.png",
                matching_accuracy=matching.get("K-Means"))
    add_cluster("K-Means++", "K-Means with spread-out centroid initialization to improve starting positions.",
                "K_Means_K++Means_Elbow_Silhoute_M4_Outputs/KMeansPlusPlus/kmeans_plus_plus_metrics.csv",
                "K_Means_K++Means_Elbow_Silhoute_M4_Outputs/KMeansPlusPlus/kmeans_plus_plus_PCA_clustering.png",
                matching_accuracy=matching.get("K-Means++"))
    add_cluster("Agglomerative clustering", "Builds nested groups by repeatedly merging nearby clusters.",
                "GlucoTrend_Module_4_Outputs/hierarchical_metrics.csv", "GlucoTrend_Module_4_Outputs/hierarchical_clusters.png")
    add_cluster("Bisecting K-Means", "Divisive hierarchical clustering that recursively splits groups with K-Means.",
                "GlucoTrend_Module_4_Outputs/divisive_metrics.csv", "GlucoTrend_Module_4_Outputs/divisive_clusters.png")
    add_cluster("DBSCAN", "Finds dense groups without a preset cluster count and labels isolated points as noise.",
                "GlucoTrend_Module_4_Outputs/dbscan_metrics.csv", "GlucoTrend_Module_4_Outputs/dbscan_clusters.png")

    anomaly = _read_first_row("GlucoTrend_Module_4_Outputs/isolation_forest_summary.csv")
    catalog.append({"name": "Isolation Forest", "family": "Anomaly detection", "description": "Flags unusual readings by isolating observations that are easier to separate from the rest.", "module": "M4",
                    "accuracy": None, "precision": None, "recall": None, "f1": None, "cluster_accuracy": None,
                    "silhouette": None, "clusters": None, "inertia": None,
                    "anomaly_rate": float(anomaly["anomaly_rate_pct"]) if anomaly else None, "variance_2d": None,
                    "mae": None, "rmse": None, "r2": None, "mape": None, "evaluation_data": None,
                    "sample_size": int(anomaly["records"]) if anomaly else None,
                    "status": "Evaluated" if anomaly else "Not run", "artifact": "GlucoTrend_Module_4_Outputs/isolation_forest_counts.png"})

    variance_path = os.path.join(OUTPUTS_DIR, "GlucoTrend_Module_4_Outputs/pca_explained_variance.csv")
    variance_2d = None
    if os.path.isfile(variance_path):
        variance_frame = pd.read_csv(variance_path)
        if not variance_frame.empty:
            variance_2d = float(variance_frame.iloc[min(1, len(variance_frame) - 1)]["cumulative_variance"] * 100)
    catalog.append({"name": "PCA", "family": "Dimensionality reduction", "description": "Projects standardized features into principal components and reports captured variance.", "module": "M4",
                    "accuracy": None, "precision": None, "recall": None, "f1": None, "cluster_accuracy": None,
                    "silhouette": None, "clusters": None, "inertia": None, "anomaly_rate": None,
                    "variance_2d": variance_2d, "sample_size": None, "mae": None, "rmse": None, "r2": None, "mape": None, "evaluation_data": None,
                    "status": "Evaluated" if variance_2d is not None else "Not run",
                    "artifact": "GlucoTrend_Module_4_Outputs/pca_cumulative_variance.png"})

    tsne_path = os.path.join(OUTPUTS_DIR, "GlucoTrend_Module_4_Outputs/tsne_embedding.csv")
    tsne_rows = len(pd.read_csv(tsne_path)) if os.path.isfile(tsne_path) else None
    catalog.append({"name": "t-SNE", "family": "Dimensionality reduction", "description": "Creates a two-dimensional embedding for visual exploration; the project does not export a quality score.", "module": "M4",
                    "accuracy": None, "precision": None, "recall": None, "f1": None, "cluster_accuracy": None,
                    "silhouette": None, "clusters": None, "inertia": None, "anomaly_rate": None,
                    "variance_2d": None, "sample_size": tsne_rows, "mae": None, "rmse": None, "r2": None, "mape": None, "evaluation_data": None,
                    "status": "Embedding available" if tsne_rows else "Not run",
                    "artifact": "GlucoTrend_Module_4_Outputs/tsne_glucose_embedding.png"})
    cv_scores = _classifier_scores("CV_Decision_Tree_M5_Outputs/cross_validation_metrics.csv")
    catalog.append({"name": "Decision Tree · 5-fold CV", "family": "Model evaluation", "description": "The decision tree evaluated with stratified five-fold cross-validation.", "module": "M5",
                    **cv_scores, "cluster_accuracy": None, "silhouette": None, "clusters": None, "inertia": None,
                    "anomaly_rate": None, "variance_2d": None, "sample_size": None,
                    "mae": None, "rmse": None, "r2": None, "mape": None, "evaluation_data": "5-fold CV",
                    "status": "Evaluated" if any(value is not None for value in cv_scores.values()) else "Not run",
                    "artifact": "CV_Decision_Tree_M5_Outputs/confusion_matrix.csv"})
    return catalog


def list_images(*subpath_parts):
    """Return web-servable relative paths for PNG files under outputs."""
    folder = os.path.join(OUTPUTS_DIR, *subpath_parts)
    if not os.path.isdir(folder):
        return []
    images = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for fname in sorted(files):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
                rel_path = os.path.relpath(os.path.join(root, fname), OUTPUTS_DIR).replace(os.sep, "/")
                images.append({
                    "path": rel_path,
                    "group": artifact_topic(rel_path),
                    "name": fname.rsplit(".", 1)[0].replace("_", " ").title(),
                })
    return images


def artifact_topic(path):
    """Assign generated artifacts to the course module they support."""
    value = path.lower()
    if "boxplots_correlation" in value:
        return "EDA · Exploratory data analysis"
    if "cv_decision_tree" in value or "cross_validation" in value:
        return "M5 · Model evaluation"
    if "_m2_outputs/" in value:
        return "M2 · Supervised linear models"
    if any(term in value for term in ("k_means", "glucotrend_module_4", "dbscan", "pca", "tsne", "hierarchical", "isolation")):
        return "M4 · Unsupervised learning"
    if any(term in value for term in ("decision_tree", "random_forest", "adaboost")):
        return "M3 · Tree-based models"
    if "target_variable" in value or "target_class" in value:
        return "M2 · Targets & feature preparation"
    return "M6 · ML engineering & other artifacts"


def list_reports():
    reports = []
    if not os.path.isdir(OUTPUTS_DIR):
        return reports
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        dirs.sort()
        for filename in sorted(files):
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, OUTPUTS_DIR).replace(os.sep, "/")
            reports.append({
                "label": filename.rsplit(".", 1)[0].replace("_", " ").title(),
                "path": rel_path,
                "topic": artifact_topic(rel_path),
                "kind": filename.rsplit(".", 1)[-1].upper() if "." in filename else "FILE",
                "size_kb": round(os.path.getsize(full_path) / 1024, 1),
            })
    return sorted(reports, key=lambda report: (ML_TOPIC_ORDER[report["topic"]], report["path"].lower()))


# ================================================================
# ROUTES
# ================================================================
@app.route("/")
def home():
    stats = get_dataset_stats()
    return render_template("home.html", stats=stats)


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    uploaded = request.files.get("dataset_file")
    if not uploaded or not uploaded.filename:
        flash("Choose a CSV file to upload.", "error")
        return redirect(url_for("home") + "#dataset-upload")
    original_name = secure_filename(uploaded.filename)
    if not original_name.lower().endswith(".csv"):
        flash("Please upload a CSV file.", "error")
        return redirect(url_for("home") + "#dataset-upload")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    saved_name = f"{uuid4().hex[:10]}_{original_name}"
    destination = os.path.join(UPLOADS_DIR, saved_name)
    uploaded.save(destination)
    try:
        preview = pd.read_csv(destination, nrows=5)
        if preview.empty or len(preview.columns) == 0:
            raise ValueError("The CSV has no data rows or columns.")
    except Exception:
        os.remove(destination)
        flash("That file could not be read as a non-empty CSV. Please check it and try again.", "error")
        return redirect(url_for("home") + "#dataset-upload")

    flash(f"{original_name} uploaded successfully ({len(preview.columns)} columns detected). Saved in dataset/uploads/.", "success")
    return redirect(url_for("home") + "#dataset-upload")


@app.route("/about")
def about():
    return render_template("about.html", stats=get_dataset_stats())


@app.route("/dataset")
def dataset():
    stats = get_dataset_stats()
    return render_template("dataset.html", stats=stats)


@app.route("/preprocessing")
def preprocessing():
    stages = [
        ("target_variable_engineering_M2.py", "M2 · Feature preparation", "Build five candidate target variables"),
        ("clean_missing_imputer_M2.py", "M2 · Feature preparation", "Deletion, mean/median imputation, forward fill, and indicators"),
        ("clean_label_encode_M2.py", "M2 · Feature preparation", "Encode categorical predictors with labels"),
        ("clean_one_hot_encod_M2.py", "M2 · Feature preparation", "One-hot encode categorical predictors"),
        ("clean_ordinal_encod_M2.py", "M2 · Feature preparation", "Ordinal encode exercise intensity"),
        ("clean_target_encode_M2.py", "M2 · Feature preparation", "Encode categories using target means"),
        ("clean_embedding_encode_M2.py", "M2 · Feature preparation", "Create compact numeric category representations"),
        ("clean_minmax_stand_norma_M2.py", "M2 · Feature preparation", "Standardize, min-max scale, and normalize features"),
        ("final_preprocess_M2.py", "M2 · Feature preparation", "Assemble the modeling-ready dataset"),
    ]
    output_files = {
        "target_variable_engineering_M2.py": "target_engineered_M2.csv",
        "clean_missing_imputer_M2.py": "clean_missing_imputer_M2.csv",
        "clean_label_encode_M2.py": "clean_label_encode_M2.csv",
        "clean_one_hot_encod_M2.py": "clean_one_hot_encoding_M2.csv",
        "clean_ordinal_encod_M2.py": "clean_ordinal_encode_M2.csv",
        "clean_target_encode_M2.py": "clean_target_encode_M2.csv",
        "clean_embedding_encode_M2.py": "clean_embedded_encode_M2.csv",
        "clean_minmax_stand_norma_M2.py": "clean_minmax_stand_norma_M2.csv",
        "final_preprocess_M2.py": "final_preprocess_M2.csv",
    }
    pipeline = []
    for script, topic, description in stages:
        output_file = output_files.get(script)
        status_path = os.path.join(DATASET_DIR, output_file) if output_file else None
        done = status_path is not None and os.path.exists(status_path)
        pipeline.append({"name": script.removesuffix(".py").replace("_", " "), "topic": topic,
                         "description": description, "done": done, "output_file": output_file if done else None})
    return render_template("preprocessing.html", pipeline=pipeline)


@app.route("/visualization")
def visualization():
    gallery = {}
    images = sorted(list_images(), key=lambda image: (ML_TOPIC_ORDER[image["group"]], image["path"].lower()))
    for image in images:
        gallery.setdefault(image["group"] or "Project outputs", []).append(image)
    outputs = list_reports()
    output_groups = {}
    for output in outputs:
        output_groups.setdefault(output["topic"], []).append(output)
    return render_template("visualization.html", gallery=gallery, outputs=outputs, output_groups=output_groups)


@app.route("/models")
def models():
    return render_template("models.html", models=get_model_showcase())


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    submitted_values = dict(FIELD_DEFAULTS)

    if request.method == "POST":
        row = {}
        for field in NUMERIC_FEATURES:
            raw_value = request.form.get(field, FIELD_DEFAULTS[field])
            try:
                row[field] = float(raw_value)
            except (TypeError, ValueError):
                row[field] = FIELD_DEFAULTS[field]
            submitted_values[field] = row[field]
        for field in CATEGORICAL_OPTIONS:
            value = request.form.get(field, FIELD_DEFAULTS[field])
            row[field] = value
            submitted_values[field] = value

        if _model is None:
            result = {"error": _model_error or "Model unavailable."}
        else:
            input_df = pd.DataFrame([row])
            prediction_label = _model.predict(input_df)[0]
            confidence = None
            if hasattr(_model, "predict_proba"):
                proba = _model.predict_proba(input_df)[0]
                classes = _model.named_steps["classifier"].classes_
                confidence = round(float(max(proba)) * 100, 1)
                class_probs = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
            else:
                class_probs = None
            result = {
                "label": prediction_label,
                "confidence": confidence,
                "class_probs": class_probs,
                "is_risk": prediction_label != "In_Range",
            }

    return render_template(
        "prediction.html",
        sections=FORM_SECTIONS,
        labels=FIELD_LABELS,
        options=CATEGORICAL_OPTIONS,
        ranges=FIELD_RANGES,
        values=submitted_values,
        result=result,
        model_available=_model is not None,
    )


@app.route("/dashboard")
def dashboard():
    stats = get_dataset_stats()
    balance = get_target_balance()
    metrics = get_model_metrics()
    best_model = max((m for m in metrics if m["available"]), key=lambda m: m.get("accuracy", 0), default=None)
    showcase = get_model_showcase()
    classification_models = [model for model in showcase if model["accuracy"] is not None and model["module"] in {"M2", "M3"}]
    regression_models = [model for model in showcase if model["mae"] is not None]
    return render_template("dashboard.html", stats=stats, balance=balance, metrics=classification_models,
                           regression_models=regression_models, best_model=best_model)


@app.route("/reports")
def reports():
    return render_template("reports.html", reports=list_reports())


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serve generated artifacts, downloading data and model files."""
    force_download = request.args.get("download") == "1"
    return send_from_directory(
        OUTPUTS_DIR, filename,
        as_attachment=force_download or filename.lower().endswith((".csv", ".txt", ".pkl", ".pickle", ".joblib")),
    )


@app.route("/dataset-download/<path:filename>")
def dataset_download(filename):
    """Download a generated dataset CSV by its safe basename."""
    allowed = {
        "GlucoBench_benchmark_dataset_RAW.csv", "M1_data_quality_audit.csv", "target_engineered_M2.csv",
        "clean_missing_imputer_M2.csv", "clean_label_encode_M2.csv",
        "clean_one_hot_encoding_M2.csv", "clean_ordinal_encode_M2.csv",
        "clean_target_encode_M2.csv", "clean_embedded_encode_M2.csv",
        "clean_minmax_stand_norma_M2.csv", "final_preprocess_M2.csv",
    }
    if filename not in allowed:
        abort(404)
    return send_from_directory(DATASET_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
