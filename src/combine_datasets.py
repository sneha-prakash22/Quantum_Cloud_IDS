import pandas as pd
import os

data_folder = "data"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

all_data = []

for file in files:

    file_path = os.path.join(data_folder, file)

    if os.path.exists(file_path):

        print("Loading:", file)

        data = pd.read_csv(file_path)

        data.columns = data.columns.str.strip()

        all_data.append(data)

    else:

        print("File not found:", file)

combined_data = pd.concat(all_data, ignore_index=True)

print("\n===================================")
print("COMBINED DATASET")
print("===================================")

print("Total records:", combined_data.shape[0])
print("Total features:", combined_data.shape[1])

print("\nLabels:")
print(combined_data["Label"].value_counts())

combined_data.to_csv(
    "data/combined_ids_dataset.csv",
    index=False
)

print("\nCombined dataset saved successfully!")