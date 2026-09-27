import matplotlib.pyplot as plt
import os


# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)


# ==========================================
# MODEL VALUES
# ==========================================

models = [
    "Classical ML",
    "Quantum-Inspired ML"
]

accuracy = [
    98.0877,
    98.2708
]

macro_f1 = [
    0.8702,
    0.8957
]

weighted_f1 = [
    0.9801,
    0.9780
]

features = [
    78,
    10
]


# ==========================================
# 1. ACCURACY GRAPH
# ==========================================

plt.figure()

plt.bar(models, accuracy)

plt.title("Accuracy Comparison")

plt.ylabel("Accuracy (%)")

plt.ylim(90, 100)

plt.tight_layout()

plt.savefig(
    "results/multiclass_accuracy_comparison.png"
)

plt.close()


# ==========================================
# 2. MACRO F1 GRAPH
# ==========================================

plt.figure()

plt.bar(models, macro_f1)

plt.title("Macro F1-Score Comparison")

plt.ylabel("Macro F1-Score")

plt.ylim(0.8, 1.0)

plt.tight_layout()

plt.savefig(
    "results/multiclass_macro_f1_comparison.png"
)

plt.close()


# ==========================================
# 3. WEIGHTED F1 GRAPH
# ==========================================

plt.figure()

plt.bar(models, weighted_f1)

plt.title("Weighted F1-Score Comparison")

plt.ylabel("Weighted F1-Score")

plt.ylim(0.9, 1.0)

plt.tight_layout()

plt.savefig(
    "results/multiclass_weighted_f1_comparison.png"
)

plt.close()


# ==========================================
# 4. FEATURE COUNT GRAPH
# ==========================================

plt.figure()

plt.bar(models, features)

plt.title("Feature Count Comparison")

plt.ylabel("Number of Features")

plt.tight_layout()

plt.savefig(
    "results/multiclass_feature_comparison.png"
)

plt.close()


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("GRAPHS CREATED SUCCESSFULLY")
print("===================================")

print("\nCreated files:")

print(
    "1. results/multiclass_accuracy_comparison.png"
)

print(
    "2. results/multiclass_macro_f1_comparison.png"
)

print(
    "3. results/multiclass_weighted_f1_comparison.png"
)

print(
    "4. results/multiclass_feature_comparison.png"
)

print("\nStep 45 completed successfully!")