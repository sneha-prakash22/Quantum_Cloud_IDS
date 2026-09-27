import subprocess
import sys

print("\n")
print("====================================================")
print("       QUANTUM-INSPIRED CYBER INTRUSION IDS")
print("====================================================")

steps = [
    (
        "Classical Multi-Class Model",
        "src/classical_multiclass_model.py"
    ),
    (
        "Quantum-Inspired Feature Selection",
        "src/quantum_multiclass_feature_selection.py"
    ),
    (
        "Quantum-Inspired Multi-Class Model",
        "src/quantum_multiclass_model.py"
    ),
    (
        "Model Comparison",
        "src/compare_multiclass_models.py"
    ),
    (
        "Confusion Matrix",
        "src/confusion_matrix.py"
    ),
    (
        "Intrusion Detection Test",
        "src/predict_multiclass_intrusion.py"
    ),
    (
        "Final Project Summary",
        "src/final_project_summary.py"
    )
]

for step_name, script in steps:

    print("\n")
    print("====================================================")
    print("RUNNING:", step_name)
    print("====================================================")

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:

        print("\nERROR occurred while running:")
        print(script)

        sys.exit(1)


print("\n")
print("====================================================")
print("       COMPLETE PROJECT DEMONSTRATION FINISHED")
print("====================================================")

print("\nAll project modules executed successfully!")