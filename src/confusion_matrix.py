import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# LOAD DATASET
# ==========================================

file_path = "data/balanced_ids_dataset.csv"

print("Loading dataset...")

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


# ==========================================
# CREATE X AND Y
# ==========================================

X = data[selected_features]

y = data["Label"]


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


# ==========================================
# LOAD QUANTUM-INSPIRED MODEL
# ==========================================

model = joblib.load(
    "models/quantum_multiclass_model.pkl"
)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("Making predictions...")

y_pred = model.predict(X_test)


# ==========================================
# CREATE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


# ==========================================
# CREATE GRAPH
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(
    figsize=(15, 12)
)

display.plot(
    ax=ax,
    xticks_rotation=90
)

plt.title(
    "Quantum-Inspired Multi-Class IDS Confusion Matrix"
)

plt.tight_layout()


# ==========================================
# SAVE GRAPH
# ==========================================

plt.savefig(
    "results/quantum_multiclass_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX CREATED")
print("===================================")

print(
    "File: results/quantum_multiclass_confusion_matrix.png"
)

print("\nStep 46 completed successfully!")