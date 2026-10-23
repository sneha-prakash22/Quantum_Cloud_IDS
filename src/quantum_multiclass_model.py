
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/balanced_ids_dataset.csv"
FEATURE_PATH = "models/quantum_multiclass_selected_features.txt"
MODEL_PATH = "models/quantum_multiclass_model.pkl"
METRICS_PATH = "results/quantum_multiclass_test_metrics.json"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================

print("Loading balanced dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("Dataset must contain a Label column.")

y = data["Label"].astype(str).str.strip()
X_all = data.drop(columns=["Label"])

X_all = X_all.apply(pd.to_numeric, errors="coerce")
X_all = X_all.replace([np.inf, -np.inf], np.nan)

valid = X_all.notna().all(axis=1) & y.notna() & y.ne("")
X_all = X_all.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# ==========================================
# 2. LOAD SELECTED FEATURES
# ==========================================

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    selected_features = [
        line.strip() for line in file if line.strip()
    ]

missing = [
    feature for feature in selected_features
    if feature not in X_all.columns
]

if missing:
    raise ValueError(f"Selected features not found: {missing}")

if len(selected_features) != 10:
    raise ValueError(
        f"Expected exactly 10 features, got {len(selected_features)}."
    )

X = X_all[selected_features]

print("\n===================================")
print("QUANTUM-INSPIRED MULTICLASS MODEL")
print("===================================")

print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

print("\nRecords:", len(X))
print("Original features:", X_all.shape[1])
print("Selected features:", X.shape[1])
print("Classes:", y.nunique())

# ==========================================
# 3. SPLIT DATA
# ==========================================

# Keep this split consistent with feature selection.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# ==========================================
# 5. EVALUATE ON TEST DATA
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

feature_reduction = (
    1 - X.shape[1] / X_all.shape[1]
) * 100

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")

print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print(f"Feature reduction: {feature_reduction:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, zero_division=0
    )
)

# ==========================================
# 6. SAVE TEST METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "original_feature_count": int(X_all.shape[1]),
    "selected_feature_count": int(X.shape[1]),
    "feature_reduction_percent": float(feature_reduction),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "cross_validation_macro_f1": 0.8442
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, MODEL_PATH)

print("\n===================================")
print("FILES SAVED")
print("===================================")
print("Model:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("\nModel training and evaluation completed!")
