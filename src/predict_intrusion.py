import joblib
import pandas as pd

# Load the quantum-inspired model
model = joblib.load("models/quantum_inspired_model.pkl")

# Load the selected feature names
with open("models/quantum_selected_features.txt", "r") as file:
    selected_features = [line.strip() for line in file.readlines()]

print("Quantum-inspired model loaded successfully!")

print("\nFeatures required for prediction:")
for feature in selected_features:
    print("-", feature)
# Example network traffic data
# Get values from the user
print("\nEnter network traffic values:")

values = []

for feature in selected_features:
    value = float(input(f"Enter {feature}: "))
    values.append(value)

# Create DataFrame
input_data = pd.DataFrame(
    [values],
    columns=selected_features
)

# Make prediction
prediction = model.predict(input_data)

# Convert prediction into a readable result
if prediction[0] == 0:
    result = "BENIGN"
else:
    result = "DDoS"

print("\n===================================")
print("       INTRUSION DETECTION")
print("===================================")
print("Prediction:", result)
print("===================================")