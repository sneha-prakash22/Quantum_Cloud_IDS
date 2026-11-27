
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "balanced_ids_dataset.csv"

MODEL_PATH = BASE_DIR / "models" / "cnn_multiclass_model.keras"
SCALER_PATH = BASE_DIR / "models" / "cnn_scaler.pkl"
ENCODER_PATH = BASE_DIR / "models" / "cnn_label_encoder.pkl"

METRICS_PATH = BASE_DIR / "results" / "cnn_test_metrics.json"
HISTORY_PATH = BASE_DIR / "results" / "cnn_training_history.csv"

RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 256

os.makedirs(BASE_DIR / "models", exist_ok=True)
os.makedirs(BASE_DIR / "results", exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================

print("\n===================================")
print("       1D CNN INTRUSION DETECTION")
print("===================================")

print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("Dataset must contain a Label column.")

# Preserve missing labels as missing values.
y = data["Label"].astype("string").str.strip()

# Separate input features.
X = data.drop(columns=["Label"])

# Convert all features to numeric.
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values.
X = X.replace([np.inf, -np.inf], np.nan)

# Remove invalid rows.
valid = X.notna().all(axis=1) & y.notna() & y.ne("")

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(str).reset_index(drop=True)

feature_names = X.columns.tolist()

print("Dataset shape:", X.shape)
print("Number of features:", len(feature_names))
print("Number of classes:", y.nunique())

print("\nClass distribution:")
print(y.value_counts())

# ==========================================
# 2. ENCODE CLASS LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

number_of_classes = len(encoder.classes_)

print("\nEncoded classes:")
for index, class_name in enumerate(encoder.classes_):
    print(index, ":", class_name)

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Create a separate validation set from training data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# ==========================================
# 4. FEATURE SCALING
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Limit extreme standardized values for numerical stability.
X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_val_scaled = np.clip(X_val_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)

# CNN expects: samples, sequence_length, channels.
X_train_cnn = X_train_scaled.astype(np.float32)[..., np.newaxis]
X_val_cnn = X_val_scaled.astype(np.float32)[..., np.newaxis]
X_test_cnn = X_test_scaled.astype(np.float32)[..., np.newaxis]

print("CNN input shape:", X_train_cnn.shape)

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

print("\nCalculating class weights...")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("Class weights calculated.")

# ==========================================
# 6. BUILD 1D CNN
# ==========================================

print("\nBuilding 1D CNN model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(len(feature_names), 1)
    ),

    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 7. TRAIN CNN
# ==========================================

print("\n===================================")
print("          CNN TRAINING")
print("===================================")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 8. EVALUATE ON TEST DATA
# ==========================================

print("\nEvaluating CNN on test data...")

y_prob = model.predict(
    X_test_cnn,
    verbose=0
)

y_pred = np.argmax(y_prob, axis=1)

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
    labels=np.arange(number_of_classes),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

print("\n===================================")
print("         CNN TEST RESULTS")
print("===================================")

print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(number_of_classes),
        target_names=encoder.classes_,
        zero_division=0
    )
)

# ==========================================
# 9. SAVE METRICS
# ==========================================

metrics = {
    "model": "1D CNN",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),

    "original_feature_count": int(len(feature_names)),
    "number_of_classes": int(number_of_classes),

    "training_records": int(len(X_train)),
    "validation_records": int(len(X_val)),
    "testing_records": int(len(X_test)),

    "epochs_completed": int(len(history.history["loss"])),
    "batch_size": BATCH_SIZE,
    "random_state": RANDOM_STATE,

    "classes": encoder.classes_.tolist(),
    "feature_names": feature_names,

    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# ==========================================
# 10. SAVE MODEL AND PREPROCESSING OBJECTS
# ==========================================

model.save(MODEL_PATH)

joblib.dump(scaler, SCALER_PATH)
joblib.dump(encoder, ENCODER_PATH)

# Save training history.
pd.DataFrame(history.history).to_csv(
    HISTORY_PATH,
    index=False
)

print("\n===================================")
print("          FILES SAVED")
print("===================================")

print("CNN model:", MODEL_PATH)
print("Scaler:", SCALER_PATH)
print("Label encoder:", ENCODER_PATH)
print("Metrics:", METRICS_PATH)
print("Training history:", HISTORY_PATH)

print("\nCNN training and evaluation completed!")
