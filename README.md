# Quantum-Inspired Machine Learning for Cyber Intrusion Detection

A research project exploring quantum-inspired feature selection for machine-learning-based cyber intrusion detection in enterprise cloud environments.

## Research Overview

The primary experiment investigates whether quantum-inspired feature selection can reduce the number of input features while maintaining or improving intrusion-detection performance.

The primary classifier comparison is:

- **Extra Trees — 78 features:** baseline model.
- **Extra Trees — 20 features:** model trained using the 20-feature subset selected through the quantum-inspired feature-selection process.

The project uses a quantum-inspired optimization approach; it does not execute on a quantum computer.

## Methodology

### Quantum-Inspired Feature Selection

The feature-selection experiment uses qubit-inspired probabilities derived from `sin(theta) ** 2`.

Candidate feature subsets are evaluated using a Random Forest classifier with 50 estimators and three-fold stratified cross-validation. Macro-F1 is used as the candidate fitness measure.

The selected 20-feature list is saved as an experiment artifact and subsequently used to train the primary 20-feature Extra Trees classifier.

### Primary Classifier

The primary classifier is Extra Trees with 500 estimators and random state 42.

The baseline and selected-feature models use the same train/test split and evaluation setup.

### Separate LightGBM Experiment

LightGBM feature-count experiments are separate from the primary Extra Trees comparison. Their classifier results are not used as the primary comparison against Extra Trees.

## Primary Experimental Results

| Metric | Extra Trees (78 features) | Extra Trees (20 features) |
|---|---:|---:|
| Accuracy | 97.8639% | 97.9656% |
| Macro-F1 | 86.7197% | 86.8955% |
| Weighted-F1 | 97.8525% | 97.9062% |
| Training time | 3.1241 s | 1.8433 s |
| Number of features | 78 | 20 |

The 20-feature model reduced the feature count by approximately 74.36% and training time by approximately 41.0%, while showing modest improvements in the reported evaluation metrics.

These results describe the specified experimental split and should not be interpreted as proof of performance in all network environments.

Some minority classes have very small test-set support. Their individual classification metrics should therefore be interpreted cautiously.

## Dashboard

The project includes a Streamlit research dashboard with the following pages:

- Security Overview
- Detection Lab
- Batch Analysis
- Feature Intelligence
- Model Evaluation
- Dataset Explorer
- Mininet Validation

The dashboard is an offline research prototype. It is not a live network sensor and does not continuously monitor enterprise network traffic.

## Independent Mininet Validation

A controlled benign TCP flow was generated in a Mininet environment and captured for offline analysis.

The paced capture produced one extracted flow record. The saved prediction classified that record as BENIGN, with a model confidence of 49.2%.

This is a limited functional demonstration, not a statistically meaningful independent accuracy evaluation. Only benign traffic was tested; attack traffic was not validated in this experiment. No generalization claim is made from this single flow.

The capture, extracted flow, prediction, and SHA-256 manifest are retained as supporting experiment artifacts.

## Reproducibility and Project Structure

The repository contains the dashboard, experiment scripts, evaluation results, and supporting validation artifacts.

- `src/` — research and experiment scripts.
- `data/` — datasets and selected validation evidence.
- `models/` — model-related artifacts, including the label encoder.
- `results/` — evaluation metrics, feature importance, and validation outputs.
- `reports/` — final experimental report.
- `dashboard.py` — Streamlit dashboard.
- `requirements.txt` — dashboard dependencies.

Run the dashboard from the project root using the instructions above. The separately distributed 20-feature Extra Trees model must be available locally. Automated downloading for cloud deployment is not yet configured.


## Run Locally

From the project root, create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Launch the dashboard:

```powershell
streamlit run dashboard.py
```

The dashboard requires the balanced dataset, evaluation artifacts, 20-feature Extra Trees label encoder, and trained 20-feature Extra Trees model.

The large trained model is distributed separately from ordinary Git repository files. Automated model downloading for deployment will be configured after the GitHub Release asset is published.
