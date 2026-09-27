import pandas as pd

input_file = "data/combined_ids_dataset.csv"

print("Loading combined dataset...")

data = pd.read_csv(input_file)

data.columns = data.columns.str.strip()

print("Original dataset shape:", data.shape)

print("\nOriginal class distribution:")
print(data["Label"].value_counts())


# Number of records we want from each class
samples_per_class = 5000

balanced_data = (
    data.groupby("Label", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), samples_per_class),
            random_state=42
        )
    )
    .reset_index(drop=True)
)
# Remove missing and infinite values
balanced_data = balanced_data.replace([float("inf"), float("-inf")], pd.NA)

balanced_data = balanced_data.dropna()

print("\nAfter removing missing values:")
print("Dataset shape:", balanced_data.shape)
print("Missing values:", balanced_data.isnull().sum().sum())

print("\nBalanced dataset shape:", balanced_data.shape)

print("\nNew class distribution:")
print(balanced_data["Label"].value_counts())


# Save the prepared dataset
output_file = "data/balanced_ids_dataset.csv"

balanced_data.to_csv(
    output_file,
    index=False
)

print("\nBalanced dataset saved successfully!")
print("File:", output_file)