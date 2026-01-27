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
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf

import pickle
from scipy.signal import resample

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


class TUABDataset(torch.utils.data.Dataset):
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


class TUEVDataset(torch.utils.data.Dataset):
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


class FilteredTUEVDataset(torch.utils.data.Dataset):
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
    def __init__(self, root, split: str):
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}.csv")
        
        df = pd.read_csv(self.file_path, header=None)
        self.data = df.values
        
        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor
