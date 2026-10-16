"""Logistic regression for binary in-range/out-of-range classification."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

from glucotrend_supervised_utils import load_split, save_classification_artifacts

TARGET = "target_binary"
X_train, X_test, y_train, y_test, preprocessor = load_split(
    TARGET, exclude_current_glucose=True, split_by_patient=False
)
model = Pipeline([("features", preprocessor),
                  ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))])
model.fit(X_train, y_train)
predicted = model.predict(X_test)
metrics = {
    "Accuracy": float(accuracy_score(y_test, predicted)),
    "Balanced Accuracy": float(balanced_accuracy_score(y_test, predicted)),
    "Precision": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
    "Recall": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
    "F1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
}
save_classification_artifacts("Logistic Regression", "Logistic_Regression", y_test, predicted, metrics, model)
print("Logistic Regression (target_binary) metrics:", metrics)
