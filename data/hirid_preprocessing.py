import pandas as pd
import csv
import gzip
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ORIGIN_DIRECTORY = "/data/stympopper/HIRID/data"
OUTPUT_DIRECTORY = ORIGIN_DIRECTORY + "/fivechannels"
PATH_TO_LABELS = "/data/stympopper/HIRID/labels.csv"
KEEP_CHANNELS = ["patientid", "reldatetime", "vm1", "pm41", "vm5", "vm20", "vm62"]
WINDOW_SIZE = 100


def preprocess_hirid_data(
    origin_dir=ORIGIN_DIRECTORY,
    output_dir=OUTPUT_DIRECTORY,
    keep_channels=KEEP_CHANNELS,
    window_size=WINDOW_SIZE,
    path_to_labels=PATH_TO_LABELS,
):
    labels_df = pd.read_csv(path_to_labels)
    for file in tqdm(
        glob(os.path.join(origin_dir, "*.csv")),
        total=len(glob(os.path.join(origin_dir, "*.csv"))),
        desc="Preprocessing HIRID CSV files",
    ):
        with open(file, "r") as f:
            df = pd.read_csv(f)
            # Keep only pids in label df
            df = df[df["patientid"].isin(labels_df.index())]
        df_group = df.groupby(["patientid", "reldatetime"])[keep_channels[2:]].mean()
        patient_matrices = []
        for patient_id, group in df_group.groupby(level=0):
            vals = group.values
            num_steps = vals.shape[0]

            if num_steps < window_size:
                # 1. Identify the last recorded values for this specific patient
                last_val = vals[-1, :]

                # 2. Create a padding matrix where every row is that last recorded value
                padding_rows = window_size - num_steps
                padding = np.tile(last_val, (padding_rows, 1))

                # 3. Stack the real data on top of the 'edge' padding
                padded_group = np.vstack([vals, padding])
                patient_matrices.append(padded_group)
            else:
                # If the stay is longer than window_size, we take the first window_size steps
                # This aligns with SOTA 'First 24h' windowing logic
                patient_matrices.append(vals[:window_size, :])

        output_file = os.path.join(output_dir, os.path.basename(file).replace(".csv", ".npy"))
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, np.array(patient_matrices))

if __name__ == "__main__":
    preprocess_hirid_data()