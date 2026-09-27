import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the DDoS dataset
file_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Replace infinite values
data = data.replace([np.inf, -np.inf], np.nan)

# Remove missing values
data = data.dropna()

# Convert labels into numbers
data["Label"] = data["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

print("Data loaded successfully!")
print("Number of records:", X.shape[0])
print("Number of features:", X.shape[1])
# Number of features to select
number_of_features_to_select = 10

print("\nQuantum-inspired feature selection")
print("Total available features:", X.shape[1])
print("Features we want to select:", number_of_features_to_select)

# Create a random binary feature-selection solution
np.random.seed(42)

selected_indices = np.random.choice(
    X.shape[1],
    size=number_of_features_to_select,
    replace=False
)

selected_indices = np.sort(selected_indices)

print("\nSelected feature indices:")
print(selected_indices)

print("\nSelected feature names:")

for index in selected_indices:
    print("-", X.columns[index])
# Create a small sample for fast feature evaluation
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=10000,
    random_state=42,
    stratify=y
)

print("\nSample size for feature evaluation:", X_sample.shape[0])


# Fitness function
def calculate_fitness(feature_indices):
    selected_X = X_sample.iloc[:, feature_indices]

    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
        selected_X,
        y_sample,
        test_size=0.20,
        random_state=42,
        stratify=y_sample
    )

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_small, y_train_small)

    predictions = model.predict(X_test_small)

    score = accuracy_score(y_test_small, predictions)

    return score


# Test the fitness of our selected features
fitness_score = calculate_fitness(selected_indices)

print("\nFitness score:", fitness_score)
# Quantum-inspired search
number_of_candidates = 20

best_features = selected_indices
best_fitness = fitness_score

print("\nStarting quantum-inspired search...")
print("Number of candidate solutions:", number_of_candidates)

for candidate in range(number_of_candidates):

    # Create a new random feature subset
    candidate_features = np.random.choice(
        X.shape[1],
        size=number_of_features_to_select,
        replace=False
    )

    candidate_features = np.sort(candidate_features)

    # Calculate fitness
    candidate_fitness = calculate_fitness(candidate_features)

    print(
        f"Candidate {candidate + 1}: "
        f"Fitness = {candidate_fitness:.4f}"
    )

    # Keep the better solution
    if candidate_fitness > best_fitness:
        best_fitness = candidate_fitness
        best_features = candidate_features.copy()

print("\nBest fitness:", best_fitness)

print("\nBest selected features:")

for index in best_features:
    print("-", X.columns[index])
# Save the selected feature names
selected_feature_names = X.columns[best_features].tolist()

with open("models/quantum_selected_features.txt", "w") as file:
    for feature in selected_feature_names:
        file.write(feature + "\n")

print("\nSelected features saved successfully!")