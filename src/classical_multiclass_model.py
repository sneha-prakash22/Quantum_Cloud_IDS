import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "data/balanced_ids_dataset.csv"

print("Loading dataset...")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

# Remove missing values
data = data.dropna()

# Separate features and labels
X = data.drop("Label", axis=1)
y = data["Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining classical Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("CLASSICAL RANDOM FOREST RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.4f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "models/classical_multiclass_model.pkl"
)

print("\nClassical multi-class model saved successfully!")