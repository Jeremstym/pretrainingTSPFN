import os
import sys
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    path_datasets = Path("/data/stympopper/UCR/univariate")
    dataset_names = [d.name for d in path_datasets.iterdir() if d.is_dir()]
    data_info_list = []
    for dataset in tqdm(dataset_names, desc="Processing datasets"):
        dataset_path = path_datasets / dataset
        X_train = np.load(dataset_path / "X_train.npy")
        y_train = np.load(dataset_path / "y_train.npy")
        X_test = np.load(dataset_path / "X_test.npy")
        y_test = np.load(dataset_path / "y_test.npy")
        # print(f"Dataset: {dataset}, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        data_info: dict = {
            "dataset": dataset,
            "num_train_samples": X_train.shape[0],
            "num_test_samples": X_test.shape[0],
            "num_features": X_train.shape[1],
            "sequence_length": X_train.shape[2],
            "num_classes": len(np.unique(y_train)),
        }
        # print(data_info)
        data_info_list.append(data_info)

    df = pd.DataFrame(data_info_list)
    df.to_csv("/data/stympopper/UCR/univariate_dataset_info.csv", index=False)


if __name__ == "__main__":
    main()
