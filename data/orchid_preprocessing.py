import os
import sys
import numpy as np
import scipy.signal as sgn
import pandas as pd
from glob import glob
from tqdm import tqdm
import yaml

origin_path = "/data/stympopper/ORCHID/database"
target_dir = "/data/stympopper/BenchmarkTSPFN/processed/ORCHID_processed/data"

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

if __name__ == "__main__":
    resample_signals(origin_path, target_dir)