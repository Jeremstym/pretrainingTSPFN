"""PyTorch Lightning DataModule and minimal Dataset wrapper for TSP-style data.

This module provides:
- TSPFNDataset: flexible loader for `.pt`/`.pth` files stored per-split or a single file per split.
- TSPFNDataModule: LightningDataModule with optional deterministic splitting from an `all` dataset.

Feel free to adapt `_load` in `TSPFNDataset` to match your on-disk format (CSV, one-file-per-sample, etc.).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Callable, Dict, Sequence, List, Union
from pathlib import Path
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TSPFNDataset(Dataset):
    """Minimal dataset for TSP-style tensors.

    Loading rules (defaults):
    - If `data_roots/` exists, load all `.pt`/`.pth` files inside.
    - Else if `data_roots.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
    - Otherwise the dataset is empty.
    """

    def __init__(
        self, data_roots: str, subsets: List[Path], split: str, split_ratio: float, transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.transform = transform
        self.subset_paths = subsets
        self.split = split
        self.split_ratio = split_ratio
        self.label_encoder = LabelEncoder()
        
        data_list = []
        for subset_path in self.subset_paths:
            data_ts = self._load_subset(subset_path)
            data_list.extend(data_ts)
        self.data_ts = data_list
        # self.num_classes = num_classes_list

    def _load_subset(self, subset_path: Path) -> None:
        path = os.path.join(subset_path)
        name_csv = os.path.basename(path)
        assert os.path.isfile(path), f"Dataset file not found: {path}"
        # Get number of lines in the file
        with open(path, "r") as f:
            total_lines = sum(1 for _ in f)

        list_df = []
        with tqdm(total=total_lines, desc=f"Loading {name_csv}") as pbar:
            for chunk in pd.read_csv(path, chunksize=1000, index_col=0):
                list_df.append(chunk)
                pbar.update(chunk.shape[0])

        df = pd.concat(list_df, ignore_index=False)
        # Encode labels to integers
        df.iloc[:, -1] = self.label_encoder.fit_transform(df.iloc[:, -1])
        df = pd.concat([df.iloc[:, :-1], df.iloc[:, -1]], axis=1)
        # num_classes = len(np.unique(df.iloc[:, -1]))
        # Split dataset
        indices = np.arange(len(df))
        labels = df.iloc[:, -1].values
        try:
            train_indices, val_indices = train_test_split(
                indices, train_size=self.split_ratio, random_state=42, shuffle=True, stratify=labels
            )
        except ValueError:
            print(f"Warning: Stratified split failed for {name_csv}, using non-stratified split instead.")
            train_indices, val_indices = train_test_split(
                indices, train_size=self.split_ratio, random_state=42, shuffle=True, stratify=None
            )
        if self.split == "train":
            df = df.iloc[train_indices]
        elif self.split == "val":
            df = df.iloc[val_indices]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # loaded_df = pd.read_csv(path, index_col=0)
        data_ts = df.values
        assert data_ts.ndim == 2

        if data_ts.shape[1] < 500:
            # Pad with zeros to have consistent feature size
            padding = np.zeros((data_ts.shape[0], 500 - data_ts.shape[1]))
            data_ts = np.hstack((data_ts, padding))

        if data_ts.shape[0] // 1024 > 1:
            # Split into chunks of 1024 samples
            data_ts = np.array_split(data_ts, data_ts.shape[0] // 1024)
            data_ts = [[torch.tensor(chunk, dtype=torch.float32)] for chunk in data_ts if len(chunk) == 1024]
        else:
            data_ts = []
        
        return data_ts #, [num_classes] * len(data_ts)

    def __len__(self) -> int:
        return len(self.data_ts)

    def __getitem__(self, idx: int):
        sample = self.data_ts[idx]
        print(f"Sample shape: {sample.shape}")
        if self.transform:
            sample = self.transform(sample)
        return sample

class TSPFNDataModule(pl.LightningDataModule):
    """LightningDataModule for TSP datasets.

    Parameters
    - data_roots: root directory for data
    - batch_size, num_workers, pin_memory: DataLoader args
    - transform: optional callable applied to subsets
    """

    def __init__(
        self,
        data_roots: str,
        subsets: Dict[Union[str, Subset], Union[str, Path]] = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = True,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.subsets = subsets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.seed = seed
        self.subset_list = [subset_path for _, subset_path in subsets.items()]

        self.current_dataset_idx = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        # TODO: fow now, train/val/test use the same subset. Later, we can modify to have different subsets for each.
            
        self.train_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subsets=self.subset_list,
            split="train",
            split_ratio=0.8,
            transform=self.transform,

        )
        self.val_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subsets=self.subset_list,
            split="val",
            split_ratio=0.8,
            transform=self.transform,
        )
        self.test_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subsets=self.subset_list,
            split="val",
            split_ratio=0.8,
            transform=self.transform,
        )

        return

    # def switch_to_next_dataset(self):
    #     self.current_dataset_idx += 1
    #     if self.current_dataset_idx < len(self.subset_list):
    #         self.setup()
    #         return True
    #     return False

    def _dataloader(self, dataset: Dataset, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False, batch_size=self.batch_size)


__all__ = ["TSPFNDataset", "TSPFNDataModule"]
