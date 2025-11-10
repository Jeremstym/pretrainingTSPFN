"""PyTorch Lightning DataModule and minimal Dataset wrapper for TSP-style data.

This module provides:
- TSPDataset: flexible loader for `.pt`/`.pth` files stored per-split or a single file per split.
- TSPDataModule: LightningDataModule with optional deterministic splitting from an `all` dataset.

Feel free to adapt `_load` in `TSPDataset` to match your on-disk format (CSV, one-file-per-sample, etc.).
"""

from __future__ import annotations

import os
from typing import Optional, Callable, Sequencex

import torch
import pytorch_lightning as pl
from tspfn.torch_imports import DataLoader, Dataset, random_split


class TSPDataset(Dataset):
    """Minimal dataset for TSP-style tensors.

    Loading rules (defaults):
    - If `data_dir/<split>/` exists, load all `.pt`/`.pth` files inside.
    - Else if `data_dir/<split>.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
    - Otherwise the dataset is empty.
    """

    def __init__(self, data_dir: str, split: str = "train", transform: Optional[Callable] = None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples: Sequence = []
        self._load()

    def _load(self) -> None:
        folder = os.path.join(self.data_dir, self.split)
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

        # fallback to single file per split
        path = os.path.join(self.data_dir, f"{self.split}.pt")
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
    - val_split/test_split: fractions used when splitting an `all` dataset
    - transform: optional callable applied to samples
    - seed: seed for deterministic splits
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.1,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download or prepare raw data if necessary. Left as a no-op by default."""
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets and splits. Called on every process in distributed settings."""
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return

        # Prefer single combined 'all' dataset if present
        all_ds = TSPDataset(self.data_dir, split="all", transform=self.transform)
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
