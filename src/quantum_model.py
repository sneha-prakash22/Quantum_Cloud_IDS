import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

# Read the selected feature names
with open("models/quantum_selected_features.txt", "r") as file:
    selected_features = [line.strip() for line in file.readlines()]

print("Selected features loaded:")
for feature in selected_features:
    print("-", feature)

# Create X using only selected features
X = data[selected_features]

# Create target
y = data["Label"]

print("\nData prepared successfully!")
print("Number of records:", X.shape[0])
print("Number of selected features:", X.shape[1])
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())
# Train the quantum-inspired optimized model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print("\nTraining quantum-inspired model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nQuantum-Inspired Model Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the quantum-inspired model
import joblib

joblib.dump(model, "models/quantum_inspired_model.pkl")

print("\nQuantum-inspired model saved successfully!")