import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import Sen12Landslides


class LightweightClassificationDataset(Dataset):
    """Patch-level labels and temporally averaged inputs for MobileNet."""

    def __init__(
        self,
        files,
        modality="s2",
        min_date="2015-12-03",
        use_dem=True,
        transform=None,
        optical_channels=4,
    ):
        self.base = Sen12Landslides(
            files=files,
            modality=modality,
            min_date=min_date,
            use_dem=use_dem,
            transforms=transform,
        )
        self.optical_channels = optical_channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        image = sample["img"].float().mean(dim=0)
        label = (sample["msk"] > 0).any().float()
        if image.shape[0] <= self.optical_channels:
            raise ValueError(
                f"Expected more than {self.optical_channels} channels, got {image.shape[0]}"
            )
        return {
            "s2": image[: self.optical_channels],
            "s1": image[self.optical_channels :],
            "label": label,
        }


class LightweightClassificationDataModule(LightningDataModule):
    """Data module for the local-data adaptation of the reference classifier."""

    def __init__(
        self,
        root_dir,
        dataset_cfg,
        batch_size=32,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        pin_memory=True,
        optical_channels=4,
        global_stats_path=None,
        global_stats_max_samples=None,
        global_clip=5.0,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.pin_memory = pin_memory
        self.optical_channels = optical_channels
        self.global_stats_path = Path(global_stats_path) if global_stats_path else None
        self.global_stats_max_samples = global_stats_max_samples
        self.global_clip = global_clip

    def _set_global_stats(self, mean, std):
        def visit(transform):
            if hasattr(transform, "set_stats"):
                transform.set_stats(mean, std)
            for child in getattr(transform, "transforms", []):
                visit(child)

        for transform in (self.train_transform, self.val_transform, self.test_transform):
            visit(transform)

    def _compute_global_stats(self, entries):
        source_entries = entries
        if self.global_stats_max_samples and len(entries) > self.global_stats_max_samples:
            source_entries = entries[:self.global_stats_max_samples]
        if self.global_stats_path and self.global_stats_path.exists():
            with open(self.global_stats_path) as f:
                cached = json.load(f)
            return cached["mean"], cached["std"]

        lock_path = self.global_stats_path.with_suffix(
            self.global_stats_path.suffix + ".lock"
        ) if self.global_stats_path else None
        lock_fd = None
        if lock_path:
            while True:
                try:
                    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(lock_fd)
                    lock_fd = None
                    break
                except FileExistsError:
                    if self.global_stats_path.exists():
                        with open(self.global_stats_path) as f:
                            cached = json.load(f)
                        return cached["mean"], cached["std"]
                    time.sleep(2)

        try:
            raw = Sen12Landslides(
                source_entries,
                modality=self.dataset_cfg.modality,
                min_date=self.dataset_cfg.get("min_date", "2015-12-03"),
                use_dem=self.dataset_cfg.get("dem", True),
            )
            total = total_sq = count = None
            for index in range(len(raw)):
                image = raw[index]["img"].numpy().astype("float64")
                valid = np.isfinite(image)
                values = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                values = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
                valid = valid.transpose(1, 0, 2, 3).reshape(values.shape)
                if total is None:
                    total = np.zeros(values.shape[0])
                    total_sq = np.zeros(values.shape[0])
                    count = np.zeros(values.shape[0])
                total += np.where(valid, values, 0).sum(axis=1)
                total_sq += np.where(valid, values * values, 0).sum(axis=1)
                count += valid.sum(axis=1)
            if total is None or np.any(count == 0):
                raise RuntimeError("Cannot compute global statistics from an empty training split")
            mean = total / count
            std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-12))
            if self.global_stats_path:
                self.global_stats_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.global_stats_path.with_suffix(
                    self.global_stats_path.suffix + ".tmp"
                )
                with open(temporary_path, "w") as f:
                    json.dump({"mean": mean.tolist(), "std": std.tolist(),
                               "clip": self.global_clip,
                               "source": "training split only",
                               "sample_count": len(source_entries),
                               "training_split_count": len(entries)}, f, indent=2)
                os.replace(temporary_path, self.global_stats_path)
            logging.info(
                "Computed lightweight global statistics from %d of %d training samples",
                len(source_entries), len(entries)
            )
            return mean.tolist(), std.tolist()
        finally:
            if lock_path:
                lock_path.unlink(missing_ok=True)

    def setup(self, stage=None):
        split_path = (
            self.root
            / "tasks"
            / self.dataset_cfg.task
            / self.dataset_cfg.version
            / self.dataset_cfg.modality
            / "splits.json"
        )
        with open(split_path) as f:
            splits = json.load(f)
        data_root = self.root / f"data/data_{self.dataset_cfg.version}"
        for entries in splits.values():
            for entry in entries:
                path = entry[self.dataset_cfg.modality]
                if not Path(path).is_absolute():
                    entry[self.dataset_cfg.modality] = str(data_root / path)
        mean, std = self._compute_global_stats(splits["train"])
        self._set_global_stats(mean, std)

        kwargs = {
            "modality": self.dataset_cfg.modality,
            "min_date": self.dataset_cfg.get("min_date", "2015-12-03"),
            "use_dem": self.dataset_cfg.get("dem", True),
            "optical_channels": self.optical_channels,
        }
        self.train_ds = LightweightClassificationDataset(
            splits["train"], transform=self.train_transform, **kwargs
        )
        self.val_ds = LightweightClassificationDataset(
            splits["val"], transform=self.val_transform, **kwargs
        )
        self.test_ds = LightweightClassificationDataset(
            splits["test"], transform=self.test_transform, **kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
