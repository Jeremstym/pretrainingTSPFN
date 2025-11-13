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

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


class TSPFNDataset(Dataset):
    """Minimal dataset for TSP-style tensors.

    Loading rules (defaults):
    - If `data_roots/` exists, load all `.pt`/`.pth` files inside.
    - Else if `data_roots.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
    - Otherwise the dataset is empty.
    """

    def __init__(self, data_roots: str, subset: Path, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.transform = transform
        self.subset_path = subset
        self._load()


    def _load(self) -> None:
        # folder = os.path.join(self.data_roots)
        # if os.path.isdir(folder):
        #     files = sorted(os.listdir(folder))
        #     subsets = []
        #     for fname in files:
        #         if not (fname.endswith(".pt") or fname.endswith(".pth")):
        #             continue
        #         path = os.path.join(folder, fname)
        #         try:
        #             subsets.append(torch.load(path))
        #         except Exception:
        #             # skip unreadable files
        #             continue
        #     self.subsets = subsets
        #     return

        # fallback to single file
        path = os.path.join(self.subset_path)
        assert os.path.isfile(path), f"Dataset file not found: {path}"
        loaded_df = pd.read_csv(path, index_col=0)
        self.data_ts = list(loaded_df.values)
        return

    def __len__(self) -> int:
        return len(self.data_ts)

    def __getitem__(self, idx: int):
        sample = self.data_ts[idx]
        if self.transform is not None:
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

        self.dataset: Dict[str, Dataset] = {}

        self.subset_list: Dict[Dataset] = {
            subset_name: subset_path
            for subset_name, subset_path in subsets.items()
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        # if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
        #     return
        
        self.dataset = {
            subset_name: TSPFNDataset(self.data_roots, subset_path, transform=self.transform)
            for subset_name, subset_path in self.subset_list.items()
        }

        # # Prefer single combined 'all' dataset if present
        # all_ds = TSPFNDataset(self.data_roots, transform=self.transform)
        # n_all = len(all_ds)
        # if n_all > 0:
        #     val_len = max(1, int(n_all * self.val_split)) if self.val_split > 0 else 0
        #     test_len = max(1, int(n_all * self.test_split)) if self.test_split > 0 else 0
        #     train_len = n_all - val_len - test_len
        #     if train_len <= 0:
        #         train_len = max(1, n_all - val_len - test_len)
        #     lengths = [train_len, val_len, test_len]
        #     total = sum(lengths)
        #     if total != n_all:
        #         lengths[0] += n_all - total
        #     generator = torch.Generator().manual_seed(self.seed)
        #     splits = random_split(all_ds, lengths, generator=generator)
        #     self.train_dataset, self.val_dataset, self.test_dataset = splits
        return

    def _dataloader(self, dataset: Dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return [
            self._dataloader(self.dataset[subset_name], shuffle=True)
            for subset_name in self.subsets.keys()
        ]

    def val_dataloader(self):
        return [
            self._dataloader(self.dataset[subset_name], shuffle=False)
            for subset_name in self.subsets.keys()
        ]

    def test_dataloader(self):
        return [
            self._dataloader(self.dataset[subset_name], shuffle=False)
            for subset_name in self.subsets.keys()
        ]


__all__ = ["TSPFNDataset", "TSPFNDataModule"]
