#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
import pickle
from glob import glob
from tqdm import tqdm

patient_list = set() # Use sets for O(1) lookup speed
seen_signals = set()
final_path_list = []

def main(stage="train"):
    all_files = glob(f"/data/stympopper/TUEV/edf/processed/processed_{stage}/*.pkl")

    for path in tqdm(all_files):
        patient = path.split("/")[-1].split("_")[0]
        
        if patient in patient_list:
            continue
        patient_list.add(patient)
        
        # Find all files for this specific patient
        patient_files = glob(f"/data/stympopper/TUEV/edf/processed/processed_{stage}/{patient}_*.pkl")
        
        for patient_file in patient_files:
            with open(patient_file, "rb") as f:
                sample = pickle.load(f)
                signal = sample['signal']
                
                # 1. Create a hash of the signal array
                # We use the underlying buffer for speed
                signal_hash = hashlib.sha256(signal.tobytes()).hexdigest()
                
                # 2. Check if the hash exists
                if signal_hash in seen_signals:
                    # print(f"SKIPPING duplicate signal in {patient_file}")
                    continue
                
                seen_signals.add(signal_hash)
                final_path_list.append(patient_file)

    # 1. Convert your kept paths to a set for O(1) lookup speed
    files_to_keep = set(final_path_list)

    # 2. Get the list of all files currently in that directory
    all_files = glob(f"/data/stympopper/TUEV/edf/processed/processed_{stage}/*.pkl")

    print(f"Total files found: {len(all_files)}")
    print(f"Files to keep: {len(files_to_keep)}")

    # 3. Iterate and remove
    count_deleted = 0
    for file_path in all_files:
        if file_path not in files_to_keep:
            try:
                os.remove(file_path)
                count_deleted += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

    print(f"Cleanup complete. Deleted {count_deleted} duplicate files.")

if __name__ == "__main__":
    main(stage="train")
    main(stage="eval")
