"""Linear regression baseline for next-reading glucose forecasting."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from glucotrend_supervised_utils import load_split, save_regression_artifacts

TARGET = "glucose_lead_1"
X_train, X_test, y_train, y_test, preprocessor = load_split(TARGET)
model = Pipeline([("features", preprocessor), ("regressor", LinearRegression())])
model.fit(X_train, y_train)
predicted = model.predict(X_test)
metrics = {
    "MAE": float(mean_absolute_error(y_test, predicted)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
    "R2": float(r2_score(y_test, predicted)),
    "MAPE": float(np.mean(np.abs((y_test.to_numpy() - predicted) / y_test.to_numpy())) * 100),
}
save_regression_artifacts("Linear Regression", "Linear_Regression", y_test, predicted, metrics, model)
print("Linear Regression (glucose_lead_1) metrics:", metrics)
