import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv(
    "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Remove missing and infinite values
data = data.replace([float("inf"), float("-inf")], pd.NA)
data = data.dropna()

# Load the selected feature names
with open("models/quantum_selected_features.txt", "r") as file:
    selected_features = [line.strip() for line in file.readlines()]

# Find the first DDoS record
ddos_data = data[data["Label"].str.strip() == "DDoS"]

print("Number of DDoS records found:", len(ddos_data))

# Select one DDoS record
ddos_record = ddos_data.iloc[0]

# Get only the selected features
input_data = pd.DataFrame(
    [[ddos_record[feature] for feature in selected_features]],
    columns=selected_features
)

print("\nActual label in dataset:", ddos_record["Label"])

print("\nDDoS test values:")
print(input_data)

# Load the quantum-inspired model
model = joblib.load("models/quantum_inspired_model.pkl")

# Make prediction
prediction = model.predict(input_data)

if prediction[0] == 0:
    result = "BENIGN"
else:
    result = "DDoS"

print("\n===================================")
print("       DDoS TEST RESULT")
print("===================================")
print("Actual Label:", ddos_record["Label"])
print("Predicted Label:", result)
print("===================================")