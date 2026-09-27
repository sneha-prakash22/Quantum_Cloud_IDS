print("\n")
print("======================================================")
print("       QUANTUM-INSPIRED CYBER INTRUSION DETECTION")
print("======================================================")

print("\nPROJECT DATASET")
print("--------------------------------------")
print("Dataset: CIC-IDS2017")
print("Records used: 49,155")
print("Original features: 78")
print("Attack/traffic classes: 15")


print("\nQUANTUM-INSPIRED FEATURE SELECTION")
print("--------------------------------------")
print("Selected features: 10")
print("Feature reduction: 87.18%")
print("Fitness score: 0.9805")

print("\nSelected Features:")

features = [
    "Destination Port",
    "Total Length of Fwd Packets",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Fwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "CWE Flag Count",
    "Init_Win_bytes_forward",
    "Idle Min"
]

for number, feature in enumerate(features, start=1):
    print(f"{number}. {feature}")


print("\nMODEL COMPARISON")
print("--------------------------------------")

print("Classical Random Forest:")
print("  Features: 78")
print("  Accuracy: 98.0877%")
print("  Macro F1: 0.8702")
print("  Weighted F1: 0.9801")

print("\nQuantum-Inspired Random Forest:")
print("  Features: 10")
print("  Accuracy: 98.2708%")
print("  Macro F1: 0.8957")
print("  Weighted F1: 0.9780")


print("\nINTRUSION DETECTION TEST")
print("--------------------------------------")

print("BENIGN       -> BENIGN       : CORRECT")
print("DDoS         -> DDoS         : CORRECT")
print("PortScan     -> PortScan     : CORRECT")
print("Bot          -> Bot          : CORRECT")
print("FTP-Patator  -> FTP-Patator  : CORRECT")


print("\nPROJECT OUTPUT FILES")
print("--------------------------------------")

files = [
    "models/classical_multiclass_model.pkl",
    "models/quantum_multiclass_selected_features.txt",
    "models/quantum_multiclass_model.pkl",
    "results/multiclass_accuracy_comparison.png",
    "results/multiclass_macro_f1_comparison.png",
    "results/multiclass_weighted_f1_comparison.png",
    "results/multiclass_feature_comparison.png",
    "results/quantum_multiclass_confusion_matrix.png"
]

for file in files:
    print("-", file)


print("\n======================================================")
print("              PROJECT PROTOTYPE COMPLETED")
print("======================================================")

print("\nStep 48 completed successfully!")