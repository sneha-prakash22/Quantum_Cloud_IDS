import pandas as pd
import numpy as np

file_path = "data/balanced_ids_dataset.csv"

print("Loading balanced dataset...")

data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Replace infinity values
data = data.replace([np.inf, -np.inf], np.nan)

print("\nDataset shape:")
print(data.shape)

print("\nNumber of features:")
print(data.shape[1] - 1)

print("\nClass distribution:")
print(data["Label"].value_counts())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nFirst 5 records:")
print(data.head())

print("\nDataset check completed!")