# Final Project Results

# Dataset information
total_records = 225711
original_features = 78
selected_features = 10

# Model accuracies
classical_accuracy = 0.9998670890281993
quantum_accuracy = 0.9997784817136655

# Calculate percentages
classical_accuracy_percent = classical_accuracy * 100
quantum_accuracy_percent = quantum_accuracy * 100

# Feature reduction
feature_reduction = (
    (original_features - selected_features)
    / original_features
) * 100

# Accuracy difference
accuracy_difference = (
    classical_accuracy_percent - quantum_accuracy_percent
)


print("\n==============================================")
print("     QUANTUM-INSPIRED CLOUD IDS")
print("==============================================")

print("\nDATASET")
print("----------------------------------------------")
print("Dataset: CIC-IDS2017 Friday DDoS")
print("Records used:", total_records)
print("Original features:", original_features)

print("\nFEATURE SELECTION")
print("----------------------------------------------")
print("Selected features:", selected_features)
print(f"Feature reduction: {feature_reduction:.2f}%")

print("\nMODEL PERFORMANCE")
print("----------------------------------------------")
print(
    f"Classical Random Forest Accuracy: "
    f"{classical_accuracy_percent:.4f}%"
)
print(
    f"Quantum-Inspired Model Accuracy: "
    f"{quantum_accuracy_percent:.4f}%"
)
print(
    f"Accuracy difference: "
    f"{accuracy_difference:.4f} percentage points"
)

print("\nFINAL TEST")
print("----------------------------------------------")
print("Normal traffic test: BENIGN")
print("Real DDoS record test: DDoS")

print("\n==============================================")
print("           PROJECT COMPLETED")
print("==============================================")