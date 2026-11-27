
# ============================================================
# QUANTUM-INSPIRED QPSO + LIGHTGBM
# Enterprise Cyber Intrusion Detection
# ============================================================

import os
import json
import time
import warnings

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


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATA_PATH = "data/balanced_ids_dataset.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "qpso_lightgbm_model_seed42.txt"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "qpso_lightgbm_metrics_seed42.json"
)

BEST_PARAMS_PATH = os.path.join(
    RESULTS_DIR,
    "qpso_lightgbm_best_parameters_seed42.json"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

N_PARTICLES = 10
N_ITERATIONS = 10

# QPSO contraction-expansion coefficient
BETA_MAX = 1.0
BETA_MIN = 0.5


# ============================================================
# 2. QUANTUM-INSPIRED SELECTED FEATURES
# ============================================================

SELECTED_FEATURES = [
    "Destination Port",
    "Bwd Packet Length Max",
    "Flow Packets/s",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Std",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "PSH Flag Count",
    "URG Flag Count",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Subflow Bwd Packets",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "Active Max",
    "Idle Mean",
    "Idle Std",
    "Idle Max"
]

# ============================================================
# 3. LOAD AND PREPARE DATASET
# ============================================================

def load_and_prepare_data():

    print("\nLoading CIC-IDS2017 dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Verify that all selected features are available
    missing_features = [
        feature for feature in SELECTED_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing selected features: {missing_features}"
        )

    if "Label" not in df.columns:
        raise ValueError("Label column not found.")

    # Keep only selected features and target label
    df = df[SELECTED_FEATURES + ["Label"]].copy()

    # Clean target labels
    df["Label"] = df["Label"].astype(str).str.strip()

    # Convert selected features to numeric values
    for feature in SELECTED_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Encode class labels
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["Label"])
    X = df[SELECTED_FEATURES].copy()

    print("\nDataset preparation completed.")
    print("Clean dataset shape:", df.shape)
    print("Number of selected features:", X.shape[1])
    print("Number of classes:", len(label_encoder.classes_))

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    return X, y, label_encoder


# ============================================================
# 4. SPLIT DATA INTO TRAINING, VALIDATION AND TEST SETS
# ============================================================

def split_dataset(X, y):

    print("\nSplitting dataset...")

    # First reserve the test set
    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # Split the remaining data into training and validation
    validation_fraction = VALIDATION_SIZE

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=RANDOM_STATE,
            stratify=y_train_val
        )
    )

    print("\nData split completed.")
    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

# ============================================================
# 5. QPSO HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAMETER_BOUNDS = {

    "n_estimators": (300, 1200),

    "learning_rate": (0.01, 0.15),

    "num_leaves": (15, 127),

    "max_depth": (4, 16),

    "min_child_samples": (10, 100),

    "reg_alpha": (0.0, 5.0),

    "reg_lambda": (0.0, 5.0)
}


# Number of parameters being optimized
N_DIMENSIONS = len(PARAMETER_BOUNDS)

# Lower and upper bounds for particle positions
LOWER_BOUNDS = np.array([
    bounds[0] for bounds in PARAMETER_BOUNDS.values()
])

UPPER_BOUNDS = np.array([
    bounds[1] for bounds in PARAMETER_BOUNDS.values()
])


# ============================================================
# 6. CONVERT PARTICLE POSITION INTO LIGHTGBM PARAMETERS
# ============================================================

def decode_particle(position):

    params = {}

    for index, parameter in enumerate(PARAMETER_BOUNDS):

        value = position[index]

        if parameter in [
            "n_estimators",
            "num_leaves",
            "max_depth",
            "min_child_samples"
        ]:
            value = int(round(value))

        params[parameter] = value

    # Ensure the number of leaves is compatible
    # with the selected maximum tree depth.
    maximum_leaves = 2 ** params["max_depth"]

    params["num_leaves"] = min(
        params["num_leaves"],
        maximum_leaves
    )

    return params

# ============================================================
# 7. QUANTUM-BEHAVED PARTICLE SWARM OPTIMIZATION
# ============================================================

def calculate_mean_best(personal_best_positions):

    return np.mean(
        personal_best_positions,
        axis=0
    )


def update_particle_position(
    current_position,
    personal_best,
    global_best,
    mean_best,
    beta,
    rng
):

    dimensions = len(current_position)

    # Random coefficients
    phi = rng.uniform(0.0, 1.0, dimensions)

    u = rng.uniform(
        1e-10,
        1.0,
        dimensions
    )

    # Local attractor
    local_attractor = (
        phi * personal_best
        + (1.0 - phi) * global_best
    )

    # Random direction
    direction = rng.choice(
        [-1.0, 1.0],
        size=dimensions
    )

    # Quantum-inspired position update
    new_position = (
        local_attractor
        + direction
        * beta
        * np.abs(mean_best - current_position)
        * np.log(1.0 / u)
    )

    # Keep all parameters within valid bounds
    new_position = np.clip(
        new_position,
        LOWER_BOUNDS,
        UPPER_BOUNDS
    )

    return new_position


# ============================================================
# 8. INITIALIZE QPSO PARTICLES
# ============================================================

def initialize_particles(rng):

    positions = rng.uniform(
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        size=(N_PARTICLES, N_DIMENSIONS)
    )

    return positions

# ============================================================
# 9. EVALUATE A QPSO PARTICLE USING LIGHTGBM
# ============================================================

def evaluate_particle(
    position,
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes
):

    # Convert particle position into LightGBM parameters
    params = decode_particle(position)

    # Fixed LightGBM configuration
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predict validation data
    predictions = model.predict(X_val)

    # Calculate validation macro F1
    validation_macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    return validation_macro_f1

# ============================================================
# 10. MAIN QPSO OPTIMIZATION LOOP
# ============================================================

def run_qpso(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes
):

    print("\nStarting QPSO optimization...")

    rng = np.random.default_rng(RANDOM_STATE)

    # Initialize particle positions
    positions = initialize_particles(rng)

    # Initialize personal best positions
    personal_best_positions = positions.copy()

    # Initialize fitness values
    personal_best_scores = np.full(
        N_PARTICLES,
        -np.inf
    )

    global_best_position = None
    global_best_score = -np.inf

    optimization_history = []

    start_time = time.time()

    # Evaluate initial particle population
    print("\nEvaluating initial particles...")

    for particle in range(N_PARTICLES):

        score = evaluate_particle(
            positions[particle],
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes
        )

        personal_best_scores[particle] = score

        print(
            f"Particle {particle + 1}/{N_PARTICLES} "
            f"| Macro F1: {score:.5f}"
        )

        if score > global_best_score:

            global_best_score = score

            global_best_position = (
                positions[particle].copy()
            )

    # Initialize optimization history
    optimization_history.append({
        "iteration": 0,
        "best_macro_f1": float(global_best_score)
    })

    # Main QPSO iterations
    for iteration in range(1, N_ITERATIONS + 1):

        print(
            f"\n========== QPSO ITERATION "
            f"{iteration}/{N_ITERATIONS} =========="
        )

        # Linearly decrease beta
        beta = (
            BETA_MAX
            - (BETA_MAX - BETA_MIN)
            * iteration / N_ITERATIONS
        )

        # Calculate mean best position
        mean_best = calculate_mean_best(
            personal_best_positions
        )

        # Update and evaluate every particle
        for particle in range(N_PARTICLES):

            new_position = update_particle_position(
                positions[particle],
                personal_best_positions[particle],
                global_best_position,
                mean_best,
                beta,
                rng
            )

            positions[particle] = new_position

            score = evaluate_particle(
                new_position,
                X_train,
                y_train,
                X_val,
                y_val,
                num_classes
            )

            # Update personal best
            if score > personal_best_scores[particle]:

                personal_best_scores[particle] = score

                personal_best_positions[particle] = (
                    new_position.copy()
                )

            # Update global best
            if score > global_best_score:

                global_best_score = score

                global_best_position = (
                    new_position.copy()
                )

            print(
                f"Particle {particle + 1}/{N_PARTICLES} "
                f"| Macro F1: {score:.5f} "
                f"| Best: {global_best_score:.5f}"
            )

        optimization_history.append({
            "iteration": iteration,
            "best_macro_f1": float(global_best_score)
        })

        print(
            f"Iteration {iteration} best Macro F1: "
            f"{global_best_score:.5f}"
        )

    elapsed_time = time.time() - start_time

    best_parameters = decode_particle(
        global_best_position
    )

    print("\nQPSO optimization completed.")

    print("Best validation Macro F1:", global_best_score)
    print("Best parameters:", best_parameters)
    print("Optimization time:", round(elapsed_time, 2), "seconds")

    return (
        best_parameters,
        global_best_score,
        optimization_history
    )

# ============================================================
# 11. TRAIN AND EVALUATE FINAL QPSO-LIGHTGBM MODEL
# ============================================================

def train_final_model(
    best_parameters,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    label_encoder
):

    print("\nTraining final QPSO-LightGBM model...")

    # Combine training and validation data
    X_train_final = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_final = np.concatenate(
        [y_train, y_val]
    )

    num_classes = len(label_encoder.classes_)

    # Create final LightGBM model
    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,

        **best_parameters,

        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    # Train using training + validation data
    final_model.fit(
        X_train_final,
        y_train_final
    )

    print("\nEvaluating on reserved test data...")

    # Predict test data
    y_pred = final_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    # Generate per-class classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    print("\n========== FINAL QPSO-LIGHTGBM RESULTS ==========")

    print(f"Accuracy:    {accuracy:.6f}")
    print(f"Macro F1:    {macro_f1:.6f}")
    print(f"Weighted F1: {weighted_f1:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Save trained model
    final_model.booster_.save_model(
        MODEL_PATH
    )

    # Save evaluation metrics
    metrics = {
        "model": "QPSO-Optimized LightGBM",
        "number_of_features": len(SELECTED_FEATURES),
        "selected_features": SELECTED_FEATURES,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "best_parameters": best_parameters,
        "classification_report": report
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nModel saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)

    return metrics

# ============================================================
# 12. MAIN EXECUTION FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("QUANTUM-INSPIRED QPSO + LIGHTGBM")
    print("Enterprise Cyber Intrusion Detection")
    print("=" * 60)

    # Create output directories if necessary
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Load dataset
    X, y, label_encoder = load_and_prepare_data()

    # Step 2: Split dataset
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    num_classes = len(label_encoder.classes_)

    # Step 3: Run QPSO optimization
    (
        best_parameters,
        best_validation_score,
        optimization_history
    ) = run_qpso(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes
    )

    # Step 4: Save best parameters
    with open(BEST_PARAMS_PATH, "w") as file:

        json.dump(
            {
                "best_validation_macro_f1": float(
                    best_validation_score
                ),
                "best_parameters": best_parameters
            },
            file,
            indent=4
        )

    # Step 5: Save optimization history
    history_df = pd.DataFrame(
        optimization_history
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "qpso_lightgbm_optimization_history_seed42.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # Step 6: Train and evaluate final model
    metrics = train_final_model(
        best_parameters,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        label_encoder
    )

    print("\n" + "=" * 60)
    print("QPSO-LIGHTGBM EXPERIMENT COMPLETED")
    print("=" * 60)

    print("\nBest validation Macro F1:",
          round(best_validation_score, 6))

    print("Final test accuracy:",
          round(metrics["accuracy"], 6))

    print("Final test Macro F1:",
          round(metrics["macro_f1"], 6))

    print("\nSaved experiment files:")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Best parameters:", BEST_PARAMS_PATH)
    print("Optimization history:", history_path)


# ============================================================
# 13. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


