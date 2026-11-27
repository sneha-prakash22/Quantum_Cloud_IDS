
import os
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 500

DATA_PATH = "data/balanced_ids_dataset.csv"

MODEL_PATH = "models/extra_trees_78_features_model.pkl"
ENCODER_PATH = "models/extra_trees_78_features_label_encoder.pkl"

METRICS_PATH = "results/extra_trees_78_features_metrics.json"
CONFUSION_PATH = "results/extra_trees_78_features_confusion_matrix.csv"
IMPORTANCE_PATH = "results/extra_trees_78_features_feature_importance.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------

print("Loading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

print("Original dataset shape:", data.shape)

if "Label" not in data.columns:
    raise ValueError("Label column not found in dataset.")

y = data["Label"].astype(str).str.strip()

X = data.drop(columns=["Label"]).copy()

# Convert features to numeric
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Handle infinite and missing values
X = X.replace([np.inf, -np.inf], np.nan)

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].astype(np.float32)
y = y.loc[valid_rows]

print("Clean dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

if X.shape[1] != 78:
    raise ValueError(
        f"Expected 78 features, but found {X.shape[1]}."
    )

# --------------------------------------------------
# 3. Encode class labels
# --------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index}: {class_name}")

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nDataset split:")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 5. Train Extra Trees
# --------------------------------------------------

model = ExtraTreesClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Extra Trees with 78 features...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")

# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

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

print("\n========== EXTRA TREES 78-FEATURE RESULTS ==========")
print(f"Accuracy:       {accuracy:.6f} ({accuracy * 100:.4f}%)")
print(f"Macro F1:       {macro_f1:.6f}")
print(f"Weighted F1:    {weighted_f1:.6f}")
print(f"Training time:  {training_time:.2f} seconds")

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
# 7. Save model and label encoder
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

# --------------------------------------------------
# 8. Save metrics
# --------------------------------------------------

metrics = {
    "model": "Extra Trees",
    "features": int(X.shape[1]),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "training_time_seconds": float(training_time),
    "n_estimators": N_ESTIMATORS,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "number_of_classes": int(len(label_encoder.classes_)),
    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# 9. Save confusion matrix
# --------------------------------------------------

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

cm_df.to_csv(CONFUSION_PATH, encoding="utf-8-sig")

# --------------------------------------------------
# 10. Save feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False
)

# --------------------------------------------------
# 11. Completion
# --------------------------------------------------

print("\nSaved files:")
print("Model:             ", MODEL_PATH)
print("Label encoder:     ", ENCODER_PATH)
print("Metrics:           ", METRICS_PATH)
print("Confusion matrix:  ", CONFUSION_PATH)
print("Feature importance:", IMPORTANCE_PATH)

print("\nExtra Trees 78-feature experiment completed successfully.")
