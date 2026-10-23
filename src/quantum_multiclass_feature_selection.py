import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_PATH = "data/balanced_ids_dataset.csv"
FEATURE_OUTPUT = "models/quantum_multiclass_selected_features.txt"
HISTORY_OUTPUT = "results/quantum_feature_selection_history.csv"

K = 10
SEARCH_SAMPLE_SIZE = 10000
NUMBER_OF_CANDIDATES = 30
LEARNING_RATE = 0.15
EXPLORATION_RATE = 0.15
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------------
# 1. LOAD AND CLEAN DATA
# -------------------------------
print("Loading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("The dataset must contain a 'Label' column.")

y = data["Label"].astype(str).str.strip()
X = data.drop(columns=["Label"])

# Ensure all model features are numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1) & y.notna() & y.ne("")
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

if X.shape[1] < K:
    raise ValueError(f"At least {K} features are required.")

print("Dataset shape:", X.shape)
print("Number of classes:", y.nunique())

# -------------------------------
# 2. HOLD OUT TEST DATA FIRST
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)

# Feature selection uses training data only
if len(X_train) > SEARCH_SAMPLE_SIZE:
    X_search, _, y_search, _ = train_test_split(
        X_train,
        y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
else:
    X_search, y_search = X_train, y_train

X_search = X_search.reset_index(drop=True)
y_search = y_search.reset_index(drop=True)

print("Feature-search records:", len(X_search))
print("Features available:", X.shape[1])
print("Features required:", K)

# -------------------------------
# 3. CROSS-VALIDATED FITNESS
# -------------------------------
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_SEED
)

def calculate_fitness(indices):
    """Return mean cross-validated macro F1-score."""

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_search.iloc[:, indices],
        y_search,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1
    )

    return float(np.mean(scores)), float(np.std(scores))


# -------------------------------
# 4. QUANTUM-INSPIRED BINARY SEARCH
# -------------------------------
print("\nStarting quantum-inspired feature selection...")

n_features = X.shape[1]

# Each feature has a qubit-inspired angle.
# Selection probability is based on sin(theta) squared.
initial_probability = K / n_features

theta = np.full(
    n_features,
    np.arcsin(np.sqrt(initial_probability))
)

best_features = None
best_fitness = -1.0
best_std = None
history = []
fitness_cache = {}

for iteration in range(NUMBER_OF_CANDIDATES):

    # Convert angles into feature-selection probabilities.
    probabilities = np.sin(theta) ** 2

    # Maintain exploration so that features are not
    # permanently excluded too early.
    probabilities = (
        (1 - EXPLORATION_RATE) * probabilities
        + EXPLORATION_RATE * initial_probability
    )

    probabilities = probabilities / probabilities.sum()

    # Select exactly K distinct features.
    candidate = np.sort(
        np.random.choice(
            n_features,
            size=K,
            replace=False,
            p=probabilities
        )
    )

    key = tuple(candidate)

    if key not in fitness_cache:
        score, score_std = calculate_fitness(candidate)
        fitness_cache[key] = (score, score_std)
    else:
        score, score_std = fitness_cache[key]

    # Update the global best candidate.
    if score > best_fitness:
        best_fitness = score
        best_std = score_std
        best_features = candidate.copy()

    # Update qubit-inspired angles toward the best
    # solution found so far.
    best_mask = np.zeros(n_features)
    best_mask[best_features] = 1

    target_angles = np.where(
        best_mask == 1,
        np.pi / 2,
        0.0
    )

    theta = (
        (1 - LEARNING_RATE) * theta
        + LEARNING_RATE * target_angles
    )

    history.append({
        "candidate": iteration + 1,
        "macro_f1_mean": score,
        "macro_f1_std": score_std,
        "best_macro_f1_so_far": best_fitness,
        "feature_indices": ",".join(map(str, candidate)),
        "feature_names": " | ".join(X.columns[candidate])
    })

    print(
        f"Candidate {iteration + 1:02d}/{NUMBER_OF_CANDIDATES}: "
        f"Macro F1 = {score:.4f} ± {score_std:.4f}; "
        f"Best = {best_fitness:.4f}"
    )

# -------------------------------
# 5. SAVE SELECTED FEATURES
# -------------------------------
selected_names = X.columns[best_features].tolist()

with open(FEATURE_OUTPUT, "w", encoding="utf-8") as file:
    file.write("\n".join(selected_names) + "\n")

pd.DataFrame(history).to_csv(HISTORY_OUTPUT, index=False)

print("\n======================================")
print("QUANTUM-INSPIRED FEATURE SELECTION")
print("======================================")
print("Selection method: Adaptive qubit-inspired binary search")
print("Fitness: 3-fold cross-validated macro F1-score")
print("Features selected:", len(selected_names))
print(f"Best CV macro F1: {best_fitness:.4f}")
print(f"CV standard deviation: {best_std:.4f}")

print("\nSelected 10 features:")
for number, name in enumerate(selected_names, start=1):
    print(f"{number}. {name}")

print("\nSaved selected features to:", FEATURE_OUTPUT)
print("Saved search history to:", HISTORY_OUTPUT)
print("\nFeature selection completed.")
print("The independent test set has not been used for selection.")