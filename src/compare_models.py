# Compare Classical ML and Quantum-Inspired ML

classical_accuracy = 0.9998670890281993
quantum_accuracy = 0.9997784817136655

classical_features = 78
quantum_features = 10

print("======================================")
print(" CLASSICAL vs QUANTUM-INSPIRED MODEL")
print("======================================")

print("\nClassical Random Forest:")
print("Number of features:", classical_features)
print("Accuracy:", classical_accuracy)

print("\nQuantum-Inspired Random Forest:")
print("Number of features:", quantum_features)
print("Accuracy:", quantum_accuracy)

# Calculate feature reduction
feature_reduction = (
    (classical_features - quantum_features)
    / classical_features
) * 100

# Calculate accuracy difference
accuracy_difference = (
    classical_accuracy - quantum_accuracy
) * 100

print("\n======================================")
print(" COMPARISON RESULTS")
print("======================================")

print(f"\nFeatures reduced: {feature_reduction:.2f}%")
print(f"Accuracy difference: {accuracy_difference:.4f}%")