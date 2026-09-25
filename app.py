"""
GlucoTrend - app.py

Same 10 routes as the placement_prediction reference app, but wired
to real data instead of static placeholders:
  - /dataset       reads the raw CSV and computes real stats
  - /visualization builds an image gallery from outputs/ on the fly
  - /models        reads each M3/M4 script's metrics CSVs
  - /prediction    loads the trained Decision Tree pipeline and
                    actually runs inference on submitted form data
  - /dashboard     aggregates headline numbers from the above
  - /reports       lists real downloadable report files
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
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


def list_images(*subpath_parts):
    """Return web-servable relative paths for every PNG under outputs/<subpath>."""
    folder = os.path.join(OUTPUTS_DIR, *subpath_parts)
    if not os.path.isdir(folder):
        return []
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".png"):
            rel_path = "/".join(subpath_parts + (fname,))
            images.append({"path": rel_path, "name": fname.replace("_", " ").replace(".png", "").title()})
    return images


def list_reports():
    reports = []
    candidates = [
        ("Target Variable Class Balance", ["Target_Variable_Analysis_M2", "target_class_balance_report.txt"]),
        ("Decision Tree - Metrics", ["Decision_Tree_Classifier_M3_Outputs", "metrics", "decision_tree_metrics.csv"]),
        ("Decision Tree - Classification Report", ["Decision_Tree_Classifier_M3_Outputs", "metrics", "classification_report.csv"]),
        ("Random Forest - Metrics", ["Random_Forest_Tree_M3_Outputs", "metrics", "random_forest_metrics.csv"]),
        ("Random Forest - Classification Report", ["Random_Forest_Tree_M3_Outputs", "metrics", "classification_report.csv"]),
        ("AdaBoost - Metrics", ["AdaBoost_Classifier_M3_Outputs", "metrics", "adaboost_metrics.csv"]),
        ("AdaBoost - Classification Report", ["AdaBoost_Classifier_M3_Outputs", "metrics", "classification_report.csv"]),
        ("K-Means - Cluster Metrics", ["K_Means_K++Means_Elbow_Silhoute_M4_Outputs", "KMeans", "kmeans_metrics.csv"]),
    ]
    for label, parts in candidates:
        full_path = os.path.join(OUTPUTS_DIR, *parts)
        if os.path.exists(full_path):
            reports.append({"label": label, "path": "/".join(parts), "size_kb": round(os.path.getsize(full_path) / 1024, 1)})
    return reports


# ================================================================
# ROUTES
# ================================================================
@app.route("/")
def home():
    stats = get_dataset_stats()
    return render_template("home.html", stats=stats)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset")
def dataset():
    stats = get_dataset_stats()
    return render_template("dataset.html", stats=stats)


@app.route("/preprocessing")
def preprocessing():
    stages = [
        ("Dataset_Load_identify_missing_values_M1.py", "Load raw data, audit missing values & duplicates"),
        ("Correlation_Matrix_heatmap_boxplots_M1.py", "Correlation heatmap + feature-vs-target boxplots"),
        ("target_variable_engineering_M2.py", "Build all 5 candidate target variables"),
        ("clean_missing_imputer_M2.py", "Deletion / mean / median / forward-fill / indicator"),
        ("clean_label_encode_M2.py", "Label encoding for categorical predictors"),
        ("clean_one_hot_encod_M2.py", "One-hot encoding"),
        ("clean_ordinal_encod_M2.py", "Ordinal encoding (exercise_intensity)"),
        ("clean_target_encode_M2.py", "Mean-target encoding"),
        ("clean_embedding_encode_M2.py", "Pandas-based embedding encoding"),
        ("clean_minmax_stand_norma_M2.py", "Standardization / min-max / L2 normalization"),
        ("final_preprocess_M2.py", "Assemble the modeling-ready dataset"),
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
    for script, description in stages:
        output_file = output_files.get(script)
        done = output_file is not None and os.path.exists(os.path.join(DATASET_DIR, output_file))
        pipeline.append({"script": script, "description": description, "done": done})
    return render_template("preprocessing.html", pipeline=pipeline)


@app.route("/visualization")
def visualization():
    gallery = {
        "Exploratory Data Analysis": list_images("Boxplots_correlation"),
        "Decision Tree": (
            list_images("Decision_Tree_Classifier_M3_Outputs", "confusion_matrix")
            + list_images("Decision_Tree_Classifier_M3_Outputs", "charts")
            + list_images("Decision_Tree_Classifier_M3_Outputs", "feature_importance")
            + list_images("Decision_Tree_Classifier_M3_Outputs", "decision_tree")
        ),
        "Random Forest": (
            list_images("Random_Forest_Tree_M3_Outputs", "confusion_matrix")
            + list_images("Random_Forest_Tree_M3_Outputs", "charts")
            + list_images("Random_Forest_Tree_M3_Outputs", "feature_importance")
            + list_images("Random_Forest_Tree_M3_Outputs", "random_forest_tree")
        ),
        "AdaBoost": (
            list_images("AdaBoost_Classifier_M3_Outputs", "confusion_matrix")
            + list_images("AdaBoost_Classifier_M3_Outputs", "charts")
            + list_images("AdaBoost_Classifier_M3_Outputs", "feature_importance")
            + list_images("AdaBoost_Classifier_M3_Outputs", "adaboost_base_tree")
        ),
        "K-Means Clustering": (
            list_images("K_Means_K++Means_Elbow_Silhoute_M4_Outputs", "Elbow")
            + list_images("K_Means_K++Means_Elbow_Silhoute_M4_Outputs", "Silhouette")
            + list_images("K_Means_K++Means_Elbow_Silhoute_M4_Outputs", "KMeans")
            + list_images("K_Means_K++Means_Elbow_Silhoute_M4_Outputs", "KMeansPlusPlus")
        ),
    }
    gallery = {section: images for section, images in gallery.items() if images}
    return render_template("visualization.html", gallery=gallery)


@app.route("/models")
def models():
    metrics = get_model_metrics()
    return render_template("models.html", metrics=metrics)


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
    return render_template("dashboard.html", stats=stats, balance=balance, metrics=metrics, best_model=best_model)


@app.route("/reports")
def reports():
    return render_template("reports.html", reports=list_reports())


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serve images/CSVs/txt reports straight from the outputs/ folder."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
