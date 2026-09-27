import pandas as pd
import numpy as np

# Load the balanced dataset
file_path = "data/balanced_ids_dataset.csv"

print("Loading balanced dataset...")

data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Remove missing values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print("\n===================================")
print("QUANTUM-INSPIRED FEATURE SELECTION")
print("===================================")

print("\nDataset shape:", data.shape)

# Separate features and labels
X = data.drop("Label", axis=1)
y = data["Label"]

print("Number of records:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

print("\nClasses:")
print(y.unique())
# Number of features we want to select
number_of_features_to_select = 10

# Create a random binary feature-selection solution
np.random.seed(42)

selected_indices = np.random.choice(
    X.shape[1],
    size=number_of_features_to_select,
    replace=False
)

selected_indices = np.sort(selected_indices)

print("\n===================================")
print("BINARY FEATURE SELECTION")
print("===================================")

print("\nTotal features:", X.shape[1])

print("Features to select:", number_of_features_to_select)

print("\nSelected feature indices:")
print(selected_indices)

print("\nSelected feature names:")

for index in selected_indices:
    print("-", X.columns[index])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Use a smaller sample for faster feature selection
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=10000,
    random_state=42,
    stratify=y
)


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

    score = accuracy_score(
        y_test_small,
        predictions
    )

    return score


# Calculate fitness of our first candidate
fitness_score = calculate_fitness(selected_indices)

print("\n===================================")
print("FITNESS SCORE")
print("===================================")

print("\nSelected features:", number_of_features_to_select)

print("Fitness score:", round(fitness_score, 4))
# Number of candidate solutions to test
number_of_candidates = 20

best_features = selected_indices.copy()
best_fitness = fitness_score

print("\n===================================")
print("QUANTUM-INSPIRED SEARCH")
print("===================================")

print("\nSearching for better feature combinations...")

for candidate in range(number_of_candidates):

    # Generate a new candidate feature combination
    candidate_features = np.random.choice(
        X.shape[1],
        size=number_of_features_to_select,
        replace=False
    )

    candidate_features = np.sort(candidate_features)

    # Calculate candidate fitness
    candidate_fitness = calculate_fitness(candidate_features)

    print(
        f"Candidate {candidate + 1}: "
        f"Fitness = {candidate_fitness:.4f}"
    )

    # Keep the best candidate
    if candidate_fitness > best_fitness:

        best_fitness = candidate_fitness

        best_features = candidate_features.copy()


print("\n===================================")
print("BEST FEATURE SET")
print("===================================")

print("\nBest fitness:", round(best_fitness, 4))

print("\nBest feature indices:")
print(best_features)

print("\nBest feature names:")

for index in best_features:
    print("-", X.columns[index])
# Save the best selected feature names

selected_feature_names = X.columns[best_features].tolist()

with open(
    "models/quantum_multiclass_selected_features.txt",
    "w"
) as file:

    for feature in selected_feature_names:
        file.write(feature + "\n")

print("\n===================================")
print("FEATURE SELECTION SAVED")
print("===================================")

print("\nSelected features saved successfully!")

print(
    "File: "
    "models/quantum_multiclass_selected_features.txt"
)