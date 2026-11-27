import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/balanced_ids_dataset.csv"

MODEL_PATH = "models/xgboost_multiclass_model.json"
ENCODER_PATH = "models/xgboost_label_encoder.pkl"

METRICS_PATH = "results/xgboost_multiclass_metrics.json"
CONFUSION_PATH = "results/xgboost_confusion_matrix.csv"
IMPORTANCE_PATH = "results/xgboost_feature_importance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_and_prepare_data():

    print("\nLoading CIC-IDS2017 dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Label column not found.")

    df["Label"] = df["Label"].astype(str).str.strip()

    X = df.drop("Label", axis=1).copy()
    y = df["Label"].copy()

    # Convert features to numeric
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove incomplete rows
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    X = X.astype(np.float32)

    print("Clean dataset shape:", X.shape)
    print("Number of features:", X.shape[1])
    print("Number of classes:", y.nunique())

    return X, y


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    start_time = time.time()

    X, y = load_and_prepare_data()

    # Encode class labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ========================================================
    # XGBOOST MODEL
    # ========================================================

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

    print("\n===================================")
    print("TRAINING XGBOOST MULTICLASS MODEL")
    print("===================================")

    model.fit(X_train, y_train)

    print("\nTraining completed!")

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_))
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_df.to_csv(CONFUSION_PATH)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    # ========================================================
    # SAVE MODEL AND ENCODER
    # ========================================================

    model.save_model(MODEL_PATH)

    joblib.dump(
        label_encoder,
        ENCODER_PATH
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    elapsed_time = time.time() - start_time

    metrics = {
        "model": "XGBoost Multiclass",
        "dataset": DATA_PATH,
        "total_samples": int(len(X)),
        "number_of_features": int(X.shape[1]),
        "number_of_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "training_time_seconds": float(elapsed_time),
        "model_parameters": model.get_params()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, default=str)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n===================================")
    print("XGBOOST MULTICLASS RESULTS")
    print("===================================")

    print(f"\nAccuracy: {accuracy:.6f}")
    print(f"Accuracy (%): {accuracy * 100:.4f}%")

    print(f"Macro F1: {macro_f1:.6f}")
    print(f"Weighted F1: {weighted_f1:.6f}")

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

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    print("\nSaved files:")
    print(MODEL_PATH)
    print(ENCODER_PATH)
    print(METRICS_PATH)
    print(CONFUSION_PATH)
    print(IMPORTANCE_PATH)

    print("\nXGBoost experiment completed successfully!")


if __name__ == "__main__":
    main()
