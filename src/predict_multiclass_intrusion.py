import pandas as pd
import joblib


# ==========================================
# LOAD DATASET
# ==========================================

file_path = "data/balanced_ids_dataset.csv"

print("Loading dataset...")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

data = data.dropna()


# ==========================================
# LOAD SELECTED FEATURES
# ==========================================

with open(
    "models/quantum_multiclass_selected_features.txt",
    "r"
) as file:

    selected_features = [
        line.strip()
        for line in file.readlines()
    ]


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/quantum_multiclass_model.pkl"
)


# ==========================================
# SELECT TEST RECORDS
# ==========================================

test_classes = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "FTP-Patator"
]


print("\n===================================")
print("MULTI-CLASS INTRUSION DETECTION")
print("===================================")


# ==========================================
# TEST DIFFERENT ATTACK TYPES
# ==========================================

for attack_class in test_classes:

    records = data[
        data["Label"] == attack_class
    ]

    if len(records) == 0:
        print(
            f"\n{attack_class}: "
            "No record found."
        )
        continue

    sample = records.iloc[[0]]

    X_sample = sample[selected_features]

    prediction = model.predict(X_sample)[0]

    print("\n-----------------------------------")

    print(
        "Actual Label:",
        attack_class
    )

    print(
        "Predicted Label:",
        prediction
    )

    if prediction == attack_class:

        print("Result: CORRECT")

    else:

        print("Result: INCORRECT")


# ==========================================
# COMPLETION
# ==========================================

print("\n===================================")
print("TESTING COMPLETED")
print("===================================")

print("\nStep 47 completed successfully!")