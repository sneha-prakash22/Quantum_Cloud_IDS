
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "balanced_ids_dataset.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "models",
    "quantum_multiclass_selected_features.txt"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models",
    "quantum_inspired_lightgbm_model.txt"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "models",
    "quantum_inspired_lightgbm_encoder.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR, "results",
    "quantum_inspired_lightgbm_test_metrics.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================
# LOAD AND CLEAN DATASET
# ==========================================

print("\nLoading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH, low_memory=False)

data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("Label column not found.")

data["Label"] = data["Label"].astype(str).str.strip()

X_all = data.drop(columns=["Label"]).apply(
    pd.to_numeric, errors="coerce"
)

X_all = X_all.replace([np.inf, -np.inf], np.nan)

valid_rows = X_all.notna().all(axis=1) & data["Label"].notna()

X_all = X_all.loc[valid_rows].copy()
y = data.loc[valid_rows, "Label"].copy()

print("Dataset shape:", X_all.shape)
print("Number of classes:", y.nunique())


# ==========================================
# LOAD QUANTUM-INSPIRED FEATURES
# ==========================================

print("\nLoading quantum-inspired selected features...")

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    selected_features = [
        line.strip()
        for line in file
        if line.strip()
    ]

missing_features = [
    feature for feature in selected_features
    if feature not in X_all.columns
]

if missing_features:
    raise ValueError(
        f"Selected features missing from dataset: {missing_features}"
    )

if len(selected_features) != 10:
    raise ValueError(
        f"Expected 10 selected features, found {len(selected_features)}."
    )

X = X_all[selected_features].copy()

print("Selected features:", len(selected_features))
print(selected_features)


# ==========================================
# ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# LIGHTGBM CLASSIFIER
# ==========================================

print("\nTraining Quantum-Inspired LightGBM...")

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# EVALUATION
# ==========================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

print("\n========== QUANTUM-INSPIRED LIGHTGBM RESULTS ==========")

print(f"Accuracy:       {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL AND RESULTS
# ==========================================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

model.booster_.save_model(MODEL_PATH)

joblib.dump(label_encoder, ENCODER_PATH)

metrics = {
    "model": "Quantum-Inspired LightGBM",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "original_feature_count": int(X_all.shape[1]),
    "selected_feature_count": int(X.shape[1]),
    "number_of_classes": int(num_classes),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "selected_features": selected_features,
    "classes": label_encoder.classes_.tolist(),
    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

print("\n========== SAVED FILES ==========")
print("Model:", MODEL_PATH)
print("Encoder:", ENCODER_PATH)
print("Metrics:", METRICS_PATH)

print("\nQuantum-Inspired LightGBM training completed!")
