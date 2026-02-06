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
from typing import Dict, List, Tuple, Union, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import inf

import pickle
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader


class TUAB2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))
        
        all_x = []
        all_y = []
        
        print(f"Loading {len(self.files)} samples into RAM for TUAB 2 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["X"]).float())
                all_y.append(sample["y"])
        
        self.X = torch.stack(all_x) 
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        if self.X.shape[2] < 500:
            self.X = F.pad(self.X, (0, 500 - self.X.shape[2] - 1), "constant", 0) # New shape [Batch, Channels, 499]
            self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*499]
        else:
            raise ValueError(f"Expected signal length of 500, but got {self.X.shape[2]}. Please check the data preprocessing.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        ds = torch.cat((self.X[index], self.Y[index]), dim=-1)  # Shape [Batch, Channels*499+1]
        return ds


class TUEV2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))
        
        all_x = []
        all_y = []
        
        print(f"Loading {len(self.files)} samples into RAM for TUEV 2 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["signal"]).float())
                all_y.append(sample["label"])
        
        self.X = torch.stack(all_x) 
        self.Y = torch.tensor(all_y, dtype=torch.long) - 1  # Convert labels from 1-6 to 0-5
        self.Y = self.Y.unsqueeze(1)  # Shape [Batch, 1]

        if self.X.shape[2] < 500:
            self.X = F.pad(self.X, (0, 500 - self.X.shape[2] - 1), "constant", 0) # New shape [Batch, Channels, 499]
            self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*499]
        else:
            raise ValueError(f"Expected signal length of 500, but got {self.X.shape[2]}. Please check the data preprocessing.")


class PTB2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))
        
        all_x = []
        all_y = []
        
        print(f"Loading {len(self.files)} samples into RAM for PTB 2 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["ecg_signal_raw"]).float())
                all_y.append(sample["true_label"])
        
        self.X = torch.stack(all_x) 
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        if self.X.shape[2] < 500:
            self.X = F.pad(self.X, (0, 500 - self.X.shape[2] - 1), "constant", 0) # New shape [Batch, Channels, 499]
            self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*499]
        else:
            raise ValueError(f"Expected signal length of 500, but got {self.X.shape[2]}. Please check the data preprocessing.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        ds = torch.cat((self.X[index], self.Y[index]), dim=-1)  # Shape [Batch, Channels*499+1]
        return ds


class TSPFNMetaDataset(Dataset):
    def __init__(self, datasets: Dict, chunk_size: int =10000):
        self.chunk_size = chunk_size
        self.chunks = []

        for ds in datasets.values():
            n = len(ds)
            if n < chunk_size:
                # Optionnel : On peut ignorer ou padder les datasets trop petits
                raise ValueError(f"Dataset of size {n} is smaller than chunk size {chunk_size}. Please check the datasets or adjust the chunk size.")
            
            for i in range(0, n - chunk_size + 1, chunk_size):
                self.chunks.append(ds[i : i + chunk_size])
            
            # (Overlapping last chunk)
            if n % chunk_size != 0:
                self.chunks.append(ds[-chunk_size:])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # On retourne le bloc de 10k (X et y)
        # Supposons que y est la dernière colonne
        batch = self.chunks[idx]
        x = batch[:, :-1]
        y = batch[:, -1]
        return x, y


class TSPFNValidationDataset(Dataset):
    def __init__(self, train_datasets_list: Dict, val_datasets_list: Dict, n_support=8000, n_query=2000):
        self.n_support = n_support
        self.n_query = n_query
        self.pairs = []

        for d_train, d_val in zip(train_datasets_list.values(), val_datasets_list.values()):
            # 1. On découpe le Val en chunks de 2000 (les Query)
            n_v = len(d_val)
            # On utilise le sliding window pour ne rien perdre du Val
            indices = range(0, n_v - n_query + 1, n_query)
            
            for i in indices:
                query_chunk = d_val[i : i + n_query]
                # On stocke le chunk de val ET une référence au train complet associé
                self.pairs.append({
                    "full_train": d_train, 
                    "query_chunk": query_chunk
                })
            
            # (Overlapping last chunk for Val)
            if n_v % n_query != 0:
                self.pairs.append({"full_train": d_train, "query_chunk": d_val[-n_query:]})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        train_data = item["full_train"]
        query_chunk, query_labels = item["query_chunk"][:, :-1], item["query_chunk"][:, -1]

        # 2. Tirage aléatoire de 8000 points dans le train correspondant
        indices_sup = torch.randperm(len(train_data))[:self.n_support]
        support_chunk = train_data[indices_sup][:, :-1]
        support_labels = train_data[indices_sup][:, -1]

        output = {"support": (support_chunk, support_labels), "query": (query_chunk, query_labels)}

        return output


class TSPFNTestDataset(Dataset):
    def __init__(self, train_datasets_list: Dict, test_datasets_list: Dict, n_support=8000, n_query=2000):
        self.n_support = n_support
        self.n_query = n_query
        self.pairs = []

        for d_train, d_test in zip(train_datasets_list.values(), test_datasets_list.values()):
            # 1. On découpe le Val en chunks de 2000 (les Query)
            n_v = len(d_test)
            # On utilise le sliding window pour ne rien perdre du Val
            indices = range(0, n_v - n_query + 1, n_query)
            
            for i in indices:
                query_chunk = d_test[i : i + n_query]
                # On stocke le chunk de val ET une référence au train complet associé
                self.pairs.append({
                    "full_train": d_train, 
                    "query_chunk": query_chunk
                })
            
            # (Overlapping last chunk for Val)
            if n_v % n_query != 0:
                self.pairs.append({"full_train": d_train, "query_chunk": d_test[-n_query:]})
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        train_data = item["full_train"]
        query_chunk, query_labels = item["query_chunk"][:, :-1], item["query_chunk"][:, -1]

        # 2. Tirage aléatoire de 8000 points dans le train correspondant
        indices_sup = torch.randperm(len(train_data))[:self.n_support]
        support_chunk = train_data[indices_sup][:, :-1]
        support_labels = train_data[indices_sup][:, -1]

        output = {"support": (support_chunk, support_labels), "query": (query_chunk, query_labels)}

        return output
