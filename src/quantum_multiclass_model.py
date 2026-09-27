import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# LOAD DATASET
# ==========================================

file_path = "data/balanced_ids_dataset.csv"

print("Loading balanced dataset...")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

data = data.dropna()


# ==========================================
# LOAD SELECTED FEATURES
# ==========================================

with open(
    "models/quantum_multiclass_selected_features.txt",
    "r"
) as file:

    selected_features = [
        line.strip()
        for line in file.readlines()
    ]


print("\n===================================")
print("QUANTUM-INSPIRED MULTI-CLASS MODEL")
print("===================================")

print("\nSelected features:")

for feature in selected_features:
    print("-", feature)


# ==========================================
# CREATE X AND Y
# ==========================================

X = data[selected_features]

y = data["Label"]


print("\nNumber of records:", X.shape[0])

print("Number of selected features:", X.shape[1])

print("Number of classes:", y.nunique())


# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)

print("Testing data:", X_test.shape)


# ==========================================
# CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n===================================")
print("TRAINING MODEL")
print("===================================")

print("Training quantum-inspired model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("QUANTUM-INSPIRED MODEL RESULTS")
print("===================================")

print(
    "Selected features:",
    X.shape[1]
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy (%): {accuracy * 100:.4f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/quantum_multiclass_model.pkl"
)


print("\n===================================")
print("MODEL SAVED")
print("===================================")

print(
    "File: models/quantum_multiclass_model.pkl"
)

print("\nStep 43 completed successfully!")