# --------------------------------------------------------
# Based on LaBraM, BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/935963004/LaBraM
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import math
import time
import json
from glob import glob
from collections import defaultdict, deque
import datetime
import numpy as np
import pandas as pd
from scipy.io import arff
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf

import pickle
from scipy.signal import resample
from torch.utils.data import Dataset

standard_1020 = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


class TUABDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        # Normalize by 100
        # X = X / 100.0
        Y = int(sample["label"][0] - 1)  # make label start from 0
        X = torch.FloatTensor(X)
        return X, Y


class FilteredTUEVDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 2 * self.sampling_rate, axis=-1)
        Y = int(sample["label"] - 1)  # make label start from 0
        X = torch.FloatTensor(X)
        mask = sample["mask"]
        mask = torch.FloatTensor(mask)
        return X, Y, mask


class ECG5000Dataset(Dataset):
    def __init__(self, root, split: str, scaler=None): # Added scaler argument
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}", f"{split}.csv")

        df = pd.read_csv(self.file_path, index_col=0)
        self.data = df.values

        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1].astype(int) - 1

        self.scaler = None

        # if split == "train":
        #     self.scaler = StandardScaler()
        #     self.X = self.scaler.fit_transform(self.X)
        # else:
        #     # Use the passed scaler, or handle the case where it's missing
        #     if scaler is None:
        #         raise ValueError("A fitted scaler must be provided for the test/val split!")
        #     self.scaler = scaler
        #     self.X = self.scaler.transform(self.X)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class ESRDataset(Dataset):
    def __init__(self, root, split: str, scaler=None):  # Added scaler argument
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}", f"{split}.csv")

        df = pd.read_csv(self.file_path, index_col=0)
        self.data = df.values

        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1].astype(int) - 1  # Convert to zero-based indexing
        self.scaler = None
        
        print(f"Loaded {len(self.X)} samples for {split} split of ESR dataset.")

        # if split == "train":
        #     self.scaler = StandardScaler()
        #     self.X = self.scaler.fit_transform(self.X)
        # else:
        #     # Use the passed scaler, or handle the case where it's missing
        #     if scaler is None:
        #         raise ValueError("A fitted scaler must be provided for the test/val split!")
        #     self.scaler = scaler
        #     self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


# class ABIDEDataset(Dataset):
#     def __init__(self, root, split: str):
#         self.root = root
#         self.file_path_directory = os.path.join(self.root, f"{split}_pca")
#         self.label_file = os.path.join(self.root, "labels.csv")

#         self.all_files = glob(os.path.join(self.file_path_directory, "*.npy"))
#         print(f"Found {len(self.all_files)} files in {self.file_path_directory}")
#         self.df_labels = pd.read_csv(self.label_file, index_col=0)

#     def __len__(self):
#         return len(self.all_files)

#     def __getitem__(self, index):
#         file_path = self.all_files[index]
#         x_sample = np.load(file_path)
#         file_name = Path(file_path).stem
#         y_sample = self.df_labels.loc[int(file_name), "target"]  # Labels are already zero-based indexed

#         x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
#         y_tensor = torch.as_tensor(y_sample, dtype=torch.long)
        
#         return x_tensor, y_tensor

class ABIDEDataset(Dataset):
    def __init__(self, root, split: str):
        self.root = root
        self.split = split
        self.file_dir = os.path.join(self.root, f"{split}")
        self.label_file = os.path.join(self.root, "labels.csv")
        
        self.all_files = sorted(glob(os.path.join(self.file_dir, "*.npy")))
        self.df_labels = pd.read_csv(self.label_file, index_col=0)
        
        # Lists to hold the actual data
        self.data = []
        self.labels = []
        self.ids = []

        self._load_data()

    def _load_data(self):
        print(f"--- Loading {self.split} set into RAM ---")
        # tqdm gives you the progress bar you requested
        for f_path in tqdm(self.all_files, desc=f"Loading {self.split}"):
            file_name = Path(f_path).stem
            sub_id = int(file_name)
            
            # Load and convert
            x_sample = np.load(f_path)
            y_sample = self.df_labels.loc[sub_id, "target"]
            
            self.data.append(x_sample)
            self.labels.append(y_sample)
            self.ids.append(sub_id)

        # Convert list of arrays into a single float32 tensor
        # Shape: (N, Time, PCA_Components)
        self.data = torch.from_numpy(np.array(self.data)).float()
        print(f"Data shape after loading: {self.data.shape}")
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # O(1) complexity since data is already in memory as a tensor
        return self.data[index], self.labels[index]