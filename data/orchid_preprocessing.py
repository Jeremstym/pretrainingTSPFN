import os
import sys
import numpy as np
import scipy.signal as sgn
import pandas as pd
from glob import glob
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split

origin_path = "/data/stympopper/ORCHID/database"
target_dir = "/data/stympopper/BenchmarkTSPFN/processed/ORCHID_processed/data"
label_dir = "/data/stympopper/BenchmarkTSPFN/processed/ORCHID_processed"

label_list = ["amyloidosis", "hta", "cmi", "CMD", "healthy"]
label_map = {label: idx for idx, label in enumerate(label_list)}

label_dict = {}


def resample_signals(path, target_dir, target_sample=64):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    list_A4C = glob(path + "/[0-9]*/[0-9]*_A4C_mask.npz")
    for file in tqdm(list_A4C):
        patient_id = file.split("/")[-1].split("_")[0]
        with open(file.replace("_A4C_mask.npz", ".yaml"), "r") as f:
            yaml_data = yaml.safe_load(f)
        label = label_map[yaml_data["diagnosis"]]
        label_dict[patient_id] = label

        data = np.load(file)
        patient_ts = {}
        for feature in data.files:
            signal = data[feature]
            resampled_signal = sgn.resample(signal, target_sample)
            patient_ts[feature] = resampled_signal
        np.savez(os.path.join(target_dir, f"{patient_id}_A4C_mask.npz"), **patient_ts)

    df_label = (
        pd.DataFrame.from_dict(label_dict, orient="index", columns=["diagnosis"])
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )
    df_label.to_csv(os.path.join(target_dir, "labels.csv"), index=False)

    print(f"Resampled signals saved to {target_dir}")


def split_train_val(path, label_csv, train_ratio=0.8):
    df = pd.read_csv(label_csv)

    df_train, df_val = train_test_split(df, test_size=1 - train_ratio, random_state=42, stratify=df["diagnosis"])
    target_dir_train = os.path.join(path, "train")
    target_dir_val = os.path.join(path, "val")
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)

    for _, row in df_train.iterrows():
        patient_id = row["patient_id"]
        src_file = os.path.join(path, f"{patient_id}_A4C_mask.npz")
        dst_file = os.path.join(target_dir_train, f"{patient_id}_A4C_mask.npz")
        os.rename(src_file, dst_file)
    for _, row in df_val.iterrows():
        patient_id = row["patient_id"]
        src_file = os.path.join(path, f"{patient_id}_A4C_mask.npz")
        dst_file = os.path.join(target_dir_val, f"{patient_id}_A4C_mask.npz")
        os.rename(src_file, dst_file)

    print(f"Data split into train and val sets. Train samples: {len(df_train)}, Val samples: {len(df_val)}")

if __name__ == "__main__":
    # resample_signals(origin_path, target_dir)
    split_train_val(target_dir, os.path.join(label_dir, "labels.csv"))
