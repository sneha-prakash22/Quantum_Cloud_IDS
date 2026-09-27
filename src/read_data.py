import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load DDoS dataset
file_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("Dataset loaded successfully!")
print("Original shape:", data.shape)

print("\nOriginal label counts:")
print(data["Label"].value_counts())

# Replace infinite values with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Remove rows containing missing values
data = data.dropna()

print("\nAfter cleaning:")
print("Shape:", data.shape)

print("\nLabel counts after cleaning:")
print(data["Label"].value_counts())
# Convert labels into numbers
data["Label"] = data["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

print("\nLabels after conversion:")
print(data["Label"].value_counts())
# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
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
# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Detailed results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(model, "models/classical_random_forest.pkl")

print("\nClassical model saved successfully!")