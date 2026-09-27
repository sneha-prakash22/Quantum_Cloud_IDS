import pandas as pd
import joblib

# Load original DDoS dataset
file_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Remove empty/infinite values
data = data.replace([float("inf"), float("-inf")], float("nan"))
data = data.dropna()

# Load the CURRENT quantum-inspired model
model = joblib.load(
    "models/quantum_multiclass_model.pkl"
)

# Load the CURRENT selected features
with open(
    "models/quantum_multiclass_selected_features.txt",
    "r"
) as file:
    selected_features = [
        line.strip()
        for line in file.readlines()
    ]

# Select only DDoS records
ddos_data = data[data["Label"] == "DDoS"]

print("\n==========================================")
print("REAL DDOS TEST")
print("==========================================")

print("Number of DDoS records:", len(ddos_data))

# Take the first real DDoS record
sample = ddos_data.iloc[[0]]

# Select the 10 features used by our model
X_sample = sample[selected_features]

# Predict
prediction = model.predict(X_sample)[0]

print("\n10 FEATURE VALUES TO ENTER IN DASHBOARD:")
print("------------------------------------------")

for feature in selected_features:
    print(feature, "=", X_sample[feature].iloc[0])

print("\nActual Label:", sample["Label"].iloc[0])
print("Predicted Label:", prediction)

if prediction == "DDoS":
    print("\nSUCCESS: DDoS detected!")
else:
    print("\nThe model did not predict DDoS.")