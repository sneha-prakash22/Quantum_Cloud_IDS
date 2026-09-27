import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "data/balanced_ids_dataset.csv"

print("Loading clean dataset...")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

# Remove rows with missing values
data = data.dropna()

# Separate features and label
X = data.drop("Label", axis=1)

y = data["Label"]
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===================================")
print("TRAINING AND TESTING DATA")
print("===================================")

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nStep 40 completed successfully!")

print("\n===================================")
print("MACHINE LEARNING DATA")
print("===================================")

print("\nTotal records:", X.shape[0])

print("Number of features:", X.shape[1])

print("Number of labels:", y.shape[0])

print("\nFeature data shape:", X.shape)

print("Label data shape:", y.shape)

print("\nAttack classes:")
print(y.value_counts())

print("\nFirst 5 labels:")
print(y.head())

print("\nStep 39 completed successfully!")