"""PyTorch Lightning DataModule and minimal Dataset wrapper for TSP-style data.

This module provides:
- TSPDataset: flexible loader for `.pt`/`.pth` files stored per-split or a single file per split.
- TSPDataModule: LightningDataModule with optional deterministic splitting from an `all` dataset.

Feel free to adapt `_load` in `TSPDataset` to match your on-disk format (CSV, one-file-per-sample, etc.).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Callable, Dict, Sequence, List, Union
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


class TSPDataset(Dataset):
    """Minimal dataset for TSP-style tensors.

    Loading rules (defaults):
    - If `data_dir/` exists, load all `.pt`/`.pth` files inside.
    - Else if `data_dir.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
    - Otherwise the dataset is empty.
    """

    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples: Sequence = []
        self._load()

    def _load(self) -> None:
        folder = os.path.join(self.data_dir)
        if os.path.isdir(folder):
            files = sorted(os.listdir(folder))
            samples = []
            for fname in files:
                if not (fname.endswith(".pt") or fname.endswith(".pth")):
                    continue
                path = os.path.join(folder, fname)
                try:
                    samples.append(torch.load(path))
                except Exception:
                    # skip unreadable files
                    continue
            self.samples = samples
            return

        # fallback to single file
        path = os.path.join(self.data_dir)
        if os.path.exists(path):
            loaded = torch.load(path)
            if isinstance(loaded, (list, tuple)):
                self.samples = list(loaded)
            elif torch.is_tensor(loaded):
                # assume first dim indexes samples
                try:
                    self.samples = [loaded[i] for i in range(loaded.shape[0])]
                except Exception:
                    self.samples = [loaded]
            else:
                self.samples = [loaded]
            return

        # nothing found
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class TSPDataModule(pl.LightningDataModule):
    """LightningDataModule for TSP datasets.

    Parameters
    - data_dir: root directory for data
    - batch_size, num_workers, pin_memory: DataLoader args
    - transform: optional callable applied to samples
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
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.seed = seed

        self.datasets: Dict[Dataset] = {}

    def prepare_data(self, df: pd.DataFrame) -> None:
        """Convert dataframe into dictionary with separated labels (last column)."""
        time_series = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        self.data_dict = {'data': time_series, 'labels': labels}
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return

        # Prefer single combined 'all' dataset if present
        all_ds = TSPDataset(self.data_dir, transform=self.transform)
        n_all = len(all_ds)
        if n_all > 0:
            val_len = max(1, int(n_all * self.val_split)) if self.val_split > 0 else 0
            test_len = max(1, int(n_all * self.test_split)) if self.test_split > 0 else 0
            train_len = n_all - val_len - test_len
            if train_len <= 0:
                train_len = max(1, n_all - val_len - test_len)
            lengths = [train_len, val_len, test_len]
            total = sum(lengths)
            if total != n_all:
                lengths[0] += (n_all - total)
            generator = torch.Generator().manual_seed(self.seed)
            splits = random_split(all_ds, lengths, generator=generator)
            self.train_dataset, self.val_dataset, self.test_dataset = splits
            return

        # Otherwise load separate splits from disk
        self.train_dataset = TSPDataset(self.data_dir, split="train", transform=self.transform)
        self.val_dataset = TSPDataset(self.data_dir, split="val", transform=self.transform)
        self.test_dataset = TSPDataset(self.data_dir, split="test", transform=self.transform)

    def _dataloader(self, dataset: Dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)


__all__ = ["TSPDataset", "TSPDataModule"]
