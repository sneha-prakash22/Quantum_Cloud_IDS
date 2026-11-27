import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/balanced_ids_dataset.csv")
DETAILS_PATH = Path("results/quantum_inspired_lightgbm_feature_count_details.json")

MODEL_PATH = Path("models/xgboost_20_features_model.json")
ENCODER_PATH = Path("models/xgboost_20_features_label_encoder.pkl")
METRICS_PATH = Path("results/xgboost_20_features_metrics.json")
CONFUSION_PATH = Path("results/xgboost_20_features_confusion_matrix.csv")
IMPORTANCE_PATH = Path("results/xgboost_20_features_feature_importance.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# Load the exact 20 features from experiment results
# --------------------------------------------------

print("Loading saved 20-feature selection...")

with open(DETAILS_PATH, "r", encoding="utf-8") as file:
    experiment_details = json.load(file)

selected_details = experiment_details.get("20")

if selected_details is None:
    raise ValueError("The 20-feature experiment details were not found.")

if selected_details.get("feature_count") != 20:
    raise ValueError("The saved experiment does not contain exactly 20 features.")

selected_features = selected_details["selected_features"]

if len(selected_features) != 20:
    raise ValueError(f"Expected 20 features, found {len(selected_features)}.")

print(f"Loaded {len(selected_features)} selected features:")
for feature in selected_features:
    print(f"  - {feature}")

# --------------------------------------------------
# Load and prepare dataset
# --------------------------------------------------

print("\nLoading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("The dataset does not contain a Label column.")

missing_features = [
    feature for feature in selected_features
    if feature not in data.columns
]

if missing_features:
    raise ValueError(f"Selected features missing from dataset: {missing_features}")

y = data["Label"].astype(str).str.strip()
X = data[selected_features].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].astype(np.float32)
y = y.loc[valid_rows]

print(f"Clean dataset shape: {X.shape}")
print(f"Number of classes: {y.nunique()}")

# --------------------------------------------------
# Encode labels
# --------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index}: {class_name}")

# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nDataset split:")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# --------------------------------------------------
# Configure XGBoost
# Same parameters as the 78-feature experiment
# --------------------------------------------------

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cpu",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining XGBoost with 20 selected features...")

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

print("\n========== XGBOOST 20-FEATURE RESULTS ==========")
print(f"Accuracy:    {accuracy:.6f} ({accuracy * 100:.4f}%)")
print(f"Macro F1:    {macro_f1:.6f}")
print(f"Weighted F1: {weighted_f1:.6f}")
print(f"Training time: {training_time:.2f} seconds")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# --------------------------------------------------
# Save model, encoder and evaluation results
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

model.save_model(MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

metrics = {
    "model": "XGBoost",
    "feature_count": len(selected_features),
    "selected_features": selected_features,
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "training_time_seconds": float(training_time),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

confusion_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
confusion_df.to_csv(CONFUSION_PATH, encoding="utf-8-sig")

importance_df = pd.DataFrame({
    "feature": selected_features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(IMPORTANCE_PATH, index=False)

print("\nSaved files:")
print(f"Model:              {MODEL_PATH}")
print(f"Label encoder:      {ENCODER_PATH}")
print(f"Metrics:            {METRICS_PATH}")
print(f"Confusion matrix:   {CONFUSION_PATH}")
print(f"Feature importance: {IMPORTANCE_PATH}")

print("\nXGBoost 20-feature experiment completed successfully.")
