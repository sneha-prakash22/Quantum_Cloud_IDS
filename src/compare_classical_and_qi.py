
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/balanced_ids_dataset.csv"
FEATURE_PATH = "models/quantum_multiclass_selected_features.txt"

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

y = data["Label"].astype(str).str.strip()
X = data.drop(columns=["Label"]).apply(
    pd.to_numeric, errors="coerce"
)
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1) & y.notna() & y.ne("")
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

with open(FEATURE_PATH, encoding="utf-8") as f:
    features = [line.strip() for line in f if line.strip()]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = {}

for name, columns in [
    ("Classical RF - 78 features", X.columns.tolist()),
    ("Quantum-inspired RF - 10 features", features)
]:
    print(f"\nTraining: {name}")

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train[columns], y_train)
    pred = model.predict(X_test[columns])

    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(
            y_test, pred, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_test, pred, average="weighted", zero_division=0
        )),
        "features": len(columns)
    }

print("\n========== FAIR MODEL COMPARISON ==========")
for name, m in results.items():
    print(f"\n{name}")
    print(f"Features: {m['features']}")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Macro F1: {m['macro_f1']:.4f}")
    print(f"Weighted F1: {m['weighted_f1']:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/fair_model_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved: results/fair_model_comparison.json")
