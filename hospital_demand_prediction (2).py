"""
INT234 Predictive Analytics Project
Project: Hospital Patient Demand Prediction
Target: patients_request (weekly service-level demand)

Expected folder structure:
hospital_predictive_project/
  hospital_demand_prediction.py
  data/
    patients.csv
    services_weekly.csv
    staff.csv
    staff_schedule.csv

Install:
pip install pandas numpy matplotlib seaborn scikit-learn

Run:
python hospital_demand_prediction.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the four supplied files
patients = pd.read_csv(DATA_DIR / "patients.csv")
weekly = pd.read_csv(DATA_DIR / "services_weekly.csv")
staff = pd.read_csv(DATA_DIR / "staff.csv")
schedule = pd.read_csv(DATA_DIR / "staff_schedule.csv")

print("\\nDATASET SHAPES")
for name, frame in {
    "patients": patients, "services_weekly": weekly,
    "staff": staff, "staff_schedule": schedule
}.items():
    print(f"{name}: {frame.shape}")

# 2. Basic cleaning and validation
patients["arrival_date"] = pd.to_datetime(patients["arrival_date"], errors="coerce")
patients["departure_date"] = pd.to_datetime(patients["departure_date"], errors="coerce")
patients["length_of_stay"] = (
    patients["departure_date"] - patients["arrival_date"]
).dt.days

# Keep weekly table as the modelling table. Do not use patients_admitted
# or patients_refused as predictors because both are calculated from demand
# and bed capacity in this dataset (target leakage).
weekly["service"] = weekly["service"].astype("category")
weekly["event"] = weekly["event"].fillna("none").astype("category")

print("\\nMISSING VALUES")
print(weekly.isnull().sum())
print("\\nDUPLICATE ROWS")
print(weekly.duplicated().sum())

# 3. Exploratory data analysis
print("\\nWEEKLY DATA SUMMARY")
print(weekly.describe(include="all"))
print("\\nREQUESTS BY SERVICE")
print(weekly.groupby("service", observed=False)["patients_request"].agg(
    ["count", "mean", "median", "min", "max"]
))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))
sns.boxplot(data=weekly, x="service", y="patients_request")
plt.title("Weekly Patient Requests by Service")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "requests_by_service_boxplot.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 5))
sns.lineplot(data=weekly, x="week", y="patients_request",
             hue="service", estimator=None)
plt.title("Weekly Patient Requests by Service")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weekly_requests.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(weekly.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Weekly Numeric Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
plt.close()

# 4. Define prediction task and prevent target leakage
TARGET = "patients_request"
FEATURES = [
    "week", "month", "service", "available_beds",
    "patient_satisfaction", "staff_morale", "event"
]
categorical_features = ["service", "event"]
numeric_features = [c for c in FEATURES if c not in categorical_features]

# Chronological split: weeks 1-41 for training, weeks 42-52 for testing.
# Each service has one row per week.
train = weekly[weekly["week"] <= 41].copy()
test = weekly[weekly["week"] > 41].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

preprocessor = ColumnTransformer(transformers=[
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.04, max_depth=2, random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=42
    )
}

results = []
fitted_models = {}
predictions = {}

for model_name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    fitted_models[model_name] = pipeline
    predictions[model_name] = y_pred

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values("RMSE")
baseline_pred = np.repeat(y_train.mean(), len(y_test))
baseline = {
    "Model": "Training-mean baseline",
    "MAE": mean_absolute_error(y_test, baseline_pred),
    "RMSE": mean_squared_error(y_test, baseline_pred) ** 0.5,
    "R2": r2_score(y_test, baseline_pred)
}
print("\\nBASELINE")
print(baseline)
print("\\nMODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# 5. Select best model by lowest RMSE
best_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_models[best_name]
best_pred = predictions[best_name]
print(f"\\nSelected model: {best_name}")

prediction_table = test[["week", "month", "service", TARGET]].copy()
prediction_table["predicted_request"] = best_pred
prediction_table["absolute_error"] = np.abs(
    prediction_table[TARGET] - prediction_table["predicted_request"]
)
prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# 6. Model evaluation plots
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.75)
low = min(y_test.min(), best_pred.min())
high = max(y_test.max(), best_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual requests")
plt.ylabel("Predicted requests")
plt.title(f"Actual vs Predicted - {best_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar")
plt.ylabel("Error in number of requests")
plt.title("Model Error Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
plt.close()

# 7. Feature importance for tree-based selected model
selected_estimator = best_pipeline.named_steps["model"]
if hasattr(selected_estimator, "feature_importances_"):
    names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(
        selected_estimator.feature_importances_, index=names
    ).sort_values(ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", header=["importance"])
    plt.figure(figsize=(8, 5))
    importance.head(12).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()

# 8. Additional operational analysis
print("\\nPATIENT LEVEL SUMMARY")
print("Mean age:", round(patients["age"].mean(), 2))
print("Mean satisfaction:", round(patients["satisfaction"].mean(), 2))
print("Mean length of stay (days):", round(patients["length_of_stay"].mean(), 2))
print("\\nSTAFF ATTENDANCE BY ROLE")
print(schedule.groupby("role")["present"].mean().round(3))

# 9. Example prediction for a new service-week
example = pd.DataFrame([{
    "week": 53,
    "month": 1,
    "service": "emergency",
    "available_beds": 35,
    "patient_satisfaction": 80,
    "staff_morale": 75,
    "event": "none"
}])
example_prediction = best_pipeline.predict(example[FEATURES])[0]
print("\\nEXAMPLE FORECAST")
print(f"Estimated emergency requests for the example week: {example_prediction:.1f}")

print(f"\\nAll outputs saved in: {OUTPUT_DIR}")
