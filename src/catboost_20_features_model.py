import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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

MODEL_PATH = Path("models/catboost_20_features_model.cbm")
ENCODER_PATH = Path("models/catboost_20_features_label_encoder.pkl")
METRICS_PATH = Path("results/catboost_20_features_metrics.json")
CONFUSION_PATH = Path("results/catboost_20_features_confusion_matrix.csv")
IMPORTANCE_PATH = Path("results/catboost_20_features_feature_importance.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# Load exact 20 selected features
# --------------------------------------------------

print("Loading saved 20-feature selection...")

with open(DETAILS_PATH, "r", encoding="utf-8") as file:
    details = json.load(file)

selected_details = details.get("20")

if selected_details is None:
    raise ValueError("20-feature experiment details not found.")

selected_features = selected_details["selected_features"]

if len(selected_features) != 20:
    raise ValueError(
        f"Expected 20 features, found {len(selected_features)}."
    )

print(f"Loaded {len(selected_features)} features:")
for feature in selected_features:
    print(f"  - {feature}")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("\nLoading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("Dataset does not contain a Label column.")

missing_features = [
    feature for feature in selected_features
    if feature not in data.columns
]

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X = data[selected_features].copy()
y = data["Label"].astype(str).str.strip()

# --------------------------------------------------
# Clean data
# --------------------------------------------------

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
# Configure CatBoost
# Same parameters as the 78-feature experiment
# --------------------------------------------------

model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    task_type="CPU",
    thread_count=-1,
    verbose=100,
    allow_writing_files=False
)

# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining CatBoost with 20 selected features...")

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test).astype(int).ravel()

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

print("\n========== CATBOOST 20-FEATURE RESULTS ==========")
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
# Save model and results
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

model.save_model(str(MODEL_PATH))
joblib.dump(label_encoder, ENCODER_PATH)

metrics = {
    "model": "CatBoost",
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
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

importance_df.to_csv(IMPORTANCE_PATH, index=False)

print("\nSaved files:")
print(f"Model:              {MODEL_PATH}")
print(f"Label encoder:      {ENCODER_PATH}")
print(f"Metrics:            {METRICS_PATH}")
print(f"Confusion matrix:   {CONFUSION_PATH}")
print(f"Feature importance: {IMPORTANCE_PATH}")

print("\nCatBoost 20-feature experiment completed successfully.")
