import matplotlib.pyplot as plt

# Model results
models = ["Classical Random Forest", "Quantum-Inspired Random Forest"]

accuracies = [0.9998670890281993, 0.9997784817136655]

features = [78, 10]

# Convert accuracy to percentage
accuracy_percent = [accuracy * 100 for accuracy in accuracies]


# ==========================================
# GRAPH 1: Accuracy Comparison
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(models, accuracy_percent)

plt.ylabel("Accuracy (%)")
plt.title("Classical vs Quantum-Inspired Model Accuracy")
plt.ylim(99.9, 100.0)

for i, value in enumerate(accuracy_percent):
    plt.text(i, value, f"{value:.4f}%", ha="center", va="bottom")

plt.tight_layout()

plt.savefig("results/accuracy_comparison.png", dpi=300)

plt.show()


# ==========================================
# GRAPH 2: Feature Comparison
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(models, features)

plt.ylabel("Number of Features")
plt.title("Classical vs Quantum-Inspired Feature Count")

for i, value in enumerate(features):
    plt.text(i, value, str(value), ha="center", va="bottom")

plt.tight_layout()

plt.savefig("results/feature_comparison.png", dpi=300)

plt.show()

print("Graphs created successfully!")