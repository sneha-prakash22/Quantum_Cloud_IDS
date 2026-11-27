
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

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
    "quantum_inspired_lightgbm_optimized.txt"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "models",
    "quantum_inspired_lightgbm_optimized_encoder.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR, "results",
    "quantum_inspired_lightgbm_optimized_metrics.json"
)

TUNING_PATH = os.path.join(
    BASE_DIR, "results",
    "quantum_inspired_lightgbm_tuning.csv"
)

RANDOM_STATE = 42


# ==========================================
# LOAD AND CLEAN DATA
# ==========================================

print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH, low_memory=False)
data.columns = data.columns.str.strip()

data["Label"] = data["Label"].astype(str).str.strip()

X_all = data.drop(columns=["Label"]).apply(
    pd.to_numeric, errors="coerce"
)

X_all = X_all.replace([np.inf, -np.inf], np.nan)

valid_rows = X_all.notna().all(axis=1) & data["Label"].notna()

X_all = X_all.loc[valid_rows].copy()
y = data.loc[valid_rows, "Label"].copy()

# ==========================================
# LOAD QUANTUM-INSPIRED FEATURES
# ==========================================

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    selected_features = [
        line.strip()
        for line in file
        if line.strip()
    ]

if len(selected_features) != 10:
    raise ValueError(
        f"Expected 10 selected features, found {len(selected_features)}"
    )

missing = [
    feature for feature in selected_features
    if feature not in X_all.columns
]

if missing:
    raise ValueError(f"Missing selected features: {missing}")

X = X_all[selected_features].copy()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

num_classes = len(encoder.classes_)

print("Records:", len(X))
print("Features:", len(selected_features))
print("Classes:", num_classes)


# ==========================================
# OUTER TRAIN / TEST SPLIT
# ==========================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Validation data comes ONLY from training data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Held-out test records:", len(X_test))


# ==========================================
# HYPERPARAMETER CONFIGURATIONS
# ==========================================

configs = [
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": None
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": None
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": None
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "class_weight": None
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": None
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 40,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "class_weight": None
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced"
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 10,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "class_weight": "balanced"
    }
]


# ==========================================
# VALIDATION-BASED TUNING
# ==========================================

print("\nStarting hyperparameter optimization...")

tuning_results = []
best_model = None
best_config = None
best_accuracy = -1
best_macro_f1 = -1

for index, config in enumerate(configs, start=1):

    print(f"\nTesting configuration {index}/{len(configs)}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **config
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, predictions)

    val_macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    val_weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Macro F1: {val_macro_f1:.4f}")

    tuning_results.append({
        "configuration": index,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "validation_weighted_f1": val_weighted_f1,
        "parameters": json.dumps(config)
    })

    # Accuracy is the primary objective.
    # Macro F1 breaks ties.
    if (
        val_accuracy > best_accuracy
        or (
            np.isclose(val_accuracy, best_accuracy)
            and val_macro_f1 > best_macro_f1
        )
    ):
        best_accuracy = val_accuracy
        best_macro_f1 = val_macro_f1
        best_config = config.copy()
        best_model = model


# ==========================================
# SAVE TUNING RESULTS
# ==========================================

os.makedirs(os.path.dirname(TUNING_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

pd.DataFrame(tuning_results).to_csv(
    TUNING_PATH,
    index=False
)

print("\nBest validation accuracy:", f"{best_accuracy * 100:.2f}%")
print("Best validation Macro F1:", f"{best_macro_f1:.4f}")
print("Best parameters:", best_config)


# ==========================================
# FINAL TRAINING ON FULL TRAINING DATA
# ==========================================

print("\nRetraining selected configuration on full training data...")

final_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    **best_config
)

final_model.fit(X_train_full, y_train_full)


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating optimized model on held-out test data...")

test_predictions = final_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    test_predictions,
    labels=np.arange(num_classes),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

print("\n========== OPTIMIZED QUANTUM-INSPIRED LIGHTGBM ==========")

print(f"Accuracy:       {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(num_classes),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# SAVE OPTIMIZED MODEL AND METRICS
# ==========================================

final_model.booster_.save_model(MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

metrics = {
    "model": "Optimized Quantum-Inspired LightGBM",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "validation_accuracy": float(best_accuracy),
    "validation_macro_f1": float(best_macro_f1),
    "selected_features": selected_features,
    "selected_feature_count": len(selected_features),
    "number_of_classes": num_classes,
    "training_records": len(X_train_full),
    "validation_records": len(X_val),
    "testing_records": len(X_test),
    "random_state": RANDOM_STATE,
    "best_parameters": best_config,
    "classes": encoder.classes_.tolist(),
    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

print("\n========== SAVED FILES ==========")
print("Optimized model:", MODEL_PATH)
print("Encoder:", ENCODER_PATH)
print("Metrics:", METRICS_PATH)
print("Tuning results:", TUNING_PATH)

print("\nOptimization completed!")
