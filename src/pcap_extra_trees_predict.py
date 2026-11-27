import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "extra_trees_20_features_model.pkl"
ENCODER_PATH = ROOT / "models" / "extra_trees_20_features_label_encoder.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Extract CICFlowMeter features from a PCAP and predict intrusion classes."
    )
    parser.add_argument("--pcap", required=True, help="Input PCAP or PCAPNG file")
    parser.add_argument(
        "--flows",
        default="results/pcap_flows.csv",
        help="Extracted flow CSV output"
    )
    parser.add_argument(
        "--predictions",
        default="results/pcap_predictions.csv",
        help="Prediction CSV output"
    )
    args = parser.parse_args()

    pcap_path = Path(args.pcap).resolve()
    flows_path = (ROOT / args.flows).resolve()
    predictions_path = (ROOT / args.predictions).resolve()

    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
        raise FileNotFoundError("The trained Extra Trees model or label encoder is missing.")

    flows_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: Extracting network flows from PCAP...")

    command = [
        "pycicflowmeter",
        "-f", str(pcap_path),
        "-o", str(flows_path),
        "--quiet"
    ]

    subprocess.run(command, check=True)

    if not flows_path.is_file():
        raise RuntimeError("The flow extractor did not create its output CSV.")

    flows = pd.read_csv(flows_path)
    print(f"Extracted flows: {len(flows)}")

    if flows.empty:
        print("No completed network flows were extracted.")
        return

    print("Step 2: Loading trained Extra Trees model...")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    expected_features = list(model.feature_names_in_)

    print(f"Model expects {len(expected_features)} features.")

    print("Step 3: Preparing model input...")

    if "Fwd Header Length.1" in expected_features:
        if "Fwd Header Length" not in flows.columns:
            raise ValueError("Required source column 'Fwd Header Length' is missing.")

        flows["Fwd Header Length.1"] = flows["Fwd Header Length"]

    missing = [f for f in expected_features if f not in flows.columns]

    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    X = flows[expected_features].copy()

    for column in expected_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    valid_rows = X.notna().all(axis=1)
    invalid_count = int((~valid_rows).sum())

    if not valid_rows.any():
        raise ValueError("No flows contain all 20 valid numeric features.")

    X_valid = X.loc[valid_rows].astype(np.float32)

    print(f"Valid flows: {len(X_valid)}")
    print(f"Skipped flows with invalid features: {invalid_count}")

    print("Step 4: Predicting flow classes...")

    predictions = model.predict(X_valid)
    predicted_labels = encoder.inverse_transform(predictions.astype(int))

    probabilities = model.predict_proba(X_valid)
    confidence = probabilities.max(axis=1)

    results = flows.loc[valid_rows].copy()
    results["Predicted Label"] = predicted_labels
    results["Prediction Confidence"] = confidence

    results.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("\nPrediction summary:")
    print(results["Predicted Label"].value_counts().to_string())

    print(f"\nFlow features saved to: {flows_path}")
    print(f"Predictions saved to: {predictions_path}")
    print("\nPCAP analysis completed.")


if __name__ == "__main__":
    main()
