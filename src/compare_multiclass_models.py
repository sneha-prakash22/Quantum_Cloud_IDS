import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# LOAD DATASET
# ==========================================

file_path = "data/balanced_ids_dataset.csv"

print("Loading dataset...")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

data = data.dropna()


# ==========================================
# CREATE FULL FEATURE DATA
# ==========================================

X_full = data.drop("Label", axis=1)

y = data["Label"]


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


X_selected = data[selected_features]


# ==========================================
# SPLIT DATA
# ==========================================

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_selected, X_test_selected, _, _ = train_test_split(
    X_selected,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD MODELS
# ==========================================

classical_model = joblib.load(
    "models/classical_multiclass_model.pkl"
)

quantum_model = joblib.load(
    "models/quantum_multiclass_model.pkl"
)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nMaking predictions...")

classical_predictions = classical_model.predict(
    X_test_full
)

quantum_predictions = quantum_model.predict(
    X_test_selected
)


# ==========================================
# CALCULATE METRICS
# ==========================================

classical_accuracy = accuracy_score(
    y_test,
    classical_predictions
)

quantum_accuracy = accuracy_score(
    y_test,
    quantum_predictions
)


classical_macro_f1 = f1_score(
    y_test,
    classical_predictions,
    average="macro"
)

quantum_macro_f1 = f1_score(
    y_test,
    quantum_predictions,
    average="macro"
)


classical_weighted_f1 = f1_score(
    y_test,
    classical_predictions,
    average="weighted"
)

quantum_weighted_f1 = f1_score(
    y_test,
    quantum_predictions,
    average="weighted"
)


# ==========================================
# FEATURE REDUCTION
# ==========================================

total_features = X_full.shape[1]

selected_features_count = X_selected.shape[1]

feature_reduction = (
    (total_features - selected_features_count)
    / total_features
) * 100


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print("\nCLASSICAL MODEL")
print("------------------------------")

print("Features:", total_features)

print(
    f"Accuracy: {classical_accuracy * 100:.4f}%"
)

print(
    f"Macro F1: {classical_macro_f1:.4f}"
)

print(
    f"Weighted F1: {classical_weighted_f1:.4f}"
)


print("\nQUANTUM-INSPIRED MODEL")
print("------------------------------")

print(
    "Features:",
    selected_features_count
)

print(
    f"Accuracy: {quantum_accuracy * 100:.4f}%"
)

print(
    f"Macro F1: {quantum_macro_f1:.4f}"
)

print(
    f"Weighted F1: {quantum_weighted_f1:.4f}"
)


print("\nFEATURE REDUCTION")
print("------------------------------")

print(
    f"Feature reduction: {feature_reduction:.2f}%"
)


print("\n===================================")
print("COMPARISON COMPLETED")
print("===================================")