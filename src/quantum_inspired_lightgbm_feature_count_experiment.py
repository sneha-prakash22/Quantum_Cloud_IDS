
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/balanced_ids_dataset.csv"
EXISTING_FEATURES_PATH = "models/quantum_multiclass_selected_features.txt"

RESULTS_DIR = "results"
MODELS_DIR = "models"

FEATURE_COUNTS = [10, 15, 20, 30]
SEARCH_SAMPLE_SIZE = 10000
NUMBER_OF_CANDIDATES = 30

LEARNING_RATE = 0.15
EXPLORATION_RATE = 0.15
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

print("Loading CIC-IDS2017 dataset...")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("The dataset must contain a Label column.")

y = data["Label"].astype(str).str.strip()
X = data.drop(columns=["Label"])

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1) & y.notna() & y.ne("")

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

print("Clean dataset shape:", X.shape)
print("Number of classes:", y.nunique())
print("Total available features:", X.shape[1])


# ============================================================
# 2. CREATE ONE FIXED TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 3. PREPARE FEATURE-SEARCH DATA
# ============================================================

if len(X_train) > SEARCH_SAMPLE_SIZE:

    X_search, _, y_search, _ = train_test_split(
        X_train,
        y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train
    )

else:
    X_search = X_train.copy()
    y_search = y_train.copy()

X_search = X_search.reset_index(drop=True)
y_search = y_search.reset_index(drop=True)

print("Feature-search records:", len(X_search))


# ============================================================
# 4. LOAD EXISTING 10 SELECTED FEATURES
# ============================================================

if not os.path.exists(EXISTING_FEATURES_PATH):
    raise FileNotFoundError(
        "Existing 10-feature selection file was not found."
    )

with open(EXISTING_FEATURES_PATH, "r", encoding="utf-8") as file:
    existing_10_features = [
        line.strip()
        for line in file
        if line.strip()
    ]

if len(existing_10_features) != 10:
    raise ValueError(
        f"Expected 10 existing features, found {len(existing_10_features)}."
    )

missing = [
    feature for feature in existing_10_features
    if feature not in X.columns
]

if missing:
    raise ValueError(f"Features missing from dataset: {missing}")

print("\nExisting 10 selected features loaded successfully.")


# ============================================================
# 5. QUANTUM-INSPIRED FEATURE SELECTION
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_SEED
)


def select_quantum_inspired_features(K):
    """
    Select K features using the project's existing
    adaptive qubit-inspired binary search method.
    """

    print(f"\nSearching for the best {K} features...")

    n_features = X.shape[1]
    initial_probability = K / n_features

    rng = np.random.RandomState(RANDOM_SEED)

    theta = np.full(
        n_features,
        np.arcsin(np.sqrt(initial_probability))
    )

    best_features = None
    best_fitness = -1.0
    best_std = None

    fitness_cache = {}

    for iteration in range(NUMBER_OF_CANDIDATES):

        probabilities = np.sin(theta) ** 2

        probabilities = (
            (1 - EXPLORATION_RATE) * probabilities
            + EXPLORATION_RATE * initial_probability
        )

        probabilities = probabilities / probabilities.sum()

        candidate = np.sort(
            rng.choice(
                n_features,
                size=K,
                replace=False,
                p=probabilities
            )
        )

        key = tuple(candidate)

        if key not in fitness_cache:

            model = RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )

            scores = cross_val_score(
                model,
                X_search.iloc[:, candidate],
                y_search,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1
            )

            fitness_cache[key] = (
                float(np.mean(scores)),
                float(np.std(scores))
            )

        score, score_std = fitness_cache[key]

        if score > best_fitness:

            best_fitness = score
            best_std = score_std
            best_features = candidate.copy()

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

        print(
            f"Candidate {iteration + 1:02d}/"
            f"{NUMBER_OF_CANDIDATES}: "
            f"Macro F1 = {score:.4f} | "
            f"Best = {best_fitness:.4f}"
        )

    selected_names = X.columns[best_features].tolist()

    print(f"\nBest CV macro F1 for {K} features: {best_fitness:.4f}")
    print(f"CV standard deviation: {best_std:.4f}")

    return selected_names, best_fitness


# ============================================================
# 6. DEFINE FEATURE SETS
# ============================================================

feature_sets = {
    10: existing_10_features
}

selection_scores = {
    10: None
}

for K in [15, 20, 30]:

    selected, score = select_quantum_inspired_features(K)

    feature_sets[K] = selected
    selection_scores[K] = score


# All 78 features: no feature selection
all_features = X.columns.tolist()

feature_sets[78] = all_features
selection_scores[78] = None


# ============================================================
# 7. PREPARE LABEL ENCODER
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

joblib.dump(
    label_encoder,
    os.path.join(
        MODELS_DIR,
        "quantum_inspired_lightgbm_feature_count_encoder.pkl"
    )
)


# ============================================================
# 8. TRAIN AND EVALUATE ALL FIVE MODELS
# ============================================================

results = []
detailed_results = {}

best_model = None
best_model_features = None
best_accuracy = -1.0

for K in [10, 15, 20, 30, 78]:

    selected_features = feature_sets[K]

    print("\n" + "=" * 60)
    print(f"TRAINING LIGHTGBM WITH {K} FEATURES")
    print("=" * 60)

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),

        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=10,
        min_child_samples=40,

        reg_alpha=1.0,
        reg_lambda=1.0,

        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_train_selected,
        y_train_encoded
    )

    predictions = model.predict(X_test_selected)

    accuracy = accuracy_score(
        y_test_encoded,
        predictions
    )

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test_encoded,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    print(f"\nFeature count: {K}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    model_path = os.path.join(
        MODELS_DIR,
        f"quantum_inspired_lightgbm_features_{K}.txt"
    )

    model.booster_.save_model(model_path)

    results.append({
        "feature_count": K,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "feature_selection_cv_macro_f1": selection_scores[K],
        "model_path": model_path
    })

    detailed_results[str(K)] = {
        "feature_count": K,
        "selected_features": selected_features,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "feature_selection_cv_macro_f1": selection_scores[K],
        "classification_report": report
    }

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_features = selected_features


# ============================================================
# 9. SAVE EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
)

results_csv = os.path.join(
    RESULTS_DIR,
    "quantum_inspired_lightgbm_feature_count_comparison.csv"
)

results_json = os.path.join(
    RESULTS_DIR,
    "quantum_inspired_lightgbm_feature_count_details.json"
)

results_df.to_csv(results_csv, index=False)

with open(results_json, "w", encoding="utf-8") as file:
    json.dump(
        detailed_results,
        file,
        indent=4
    )


# ============================================================
# 10. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("QUANTUM-INSPIRED LIGHTGBM FEATURE-COUNT COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "feature_count",
            "accuracy_percent",
            "macro_f1",
            "weighted_f1"
        ]
    ].to_string(index=False)
)

print("\nBest accuracy:", f"{best_accuracy * 100:.2f}%")
print("Best feature count:", len(best_model_features))

print("\nSaved comparison:", results_csv)
print("Saved detailed results:", results_json)

print("\nAll five experiments completed successfully.")
