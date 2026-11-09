import json
import logging
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.datasets import Sen12Landslides


class Sen12LsDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        dataset_cfg,        
        batch_size,
        num_workers,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        pin_memory=True,
        normalization="precomputed",
        global_stats_path=None,
        global_clip=5.0,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.version = dataset_cfg.version
        self.task = dataset_cfg.task
        self.modality = dataset_cfg.modality.lower()
        self.use_dem = dataset_cfg.get("dem", False)
        self.min_date = dataset_cfg.get("min_date", "2015-12-03")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.pin_memory = pin_memory
        self.normalization = normalization
        self.global_stats_path = Path(global_stats_path) if global_stats_path else None
        self.global_clip = global_clip

    def _set_global_stats(self, mean, std):
        def visit(transform):
            if hasattr(transform, "set_stats"):
                transform.set_stats(mean, std)
            for child in getattr(transform, "transforms", []):
                visit(child)

        for transform in (self.train_transforms, self.val_transforms, self.test_transforms):
            visit(transform)

    def _compute_global_stats(self, entries, dataset_root):
        if self.global_stats_path and self.global_stats_path.exists():
            with open(self.global_stats_path) as f:
                cached = json.load(f)
            return cached["mean"], cached["std"]

        raw_ds = Sen12Landslides(
            files=entries,
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=None,
        )
        total = None
        total_sq = None
        count = None
        for index in range(len(raw_ds)):
            image = raw_ds[index]["img"].numpy().astype("float64")
            flat = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).transpose(1, 0, 2, 3)
            flat = flat.reshape(flat.shape[0], -1)
            valid = np.isfinite(image).transpose(1, 0, 2, 3).reshape(flat.shape)
            if total is None:
                channels = flat.shape[0]
                total = np.zeros(channels, dtype=np.float64)
                total_sq = np.zeros(channels, dtype=np.float64)
                count = np.zeros(channels, dtype=np.float64)
            total += np.where(valid, flat, 0.0).sum(axis=1)
            total_sq += np.where(valid, flat * flat, 0.0).sum(axis=1)
            count += valid.sum(axis=1)

        if total is None or np.any(count == 0):
            raise RuntimeError("Cannot compute global normalization statistics from empty data")
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-12))
        if self.global_stats_path:
            self.global_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.global_stats_path, "w") as f:
                json.dump({
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "num_samples": len(entries),
                    "clip": self.global_clip,
                    "source": "training split only",
                }, f, indent=2)
        logging.info("Computed global normalization from %d training samples", len(entries))
        return mean.tolist(), std.tolist()

    def _resolve_paths(self, splits, root):
        """Convert relative paths to absolute."""
        for split_list in splits.values():
            for entry in split_list:
                if self.modality in entry and not Path(entry[self.modality]).is_absolute():
                    entry[self.modality] = str(root / entry[self.modality])
        return splits
    
    def setup(self, stage=None):
        splits_file = self.root / "tasks" / self.task / self.version / self.modality / "splits.json"
        with open(splits_file, "r") as f:
            splits = json.load(f)
            
        dataset_root = self.root / "data" / f"data_{self.version}"
        splits = self._resolve_paths(splits, dataset_root)
        if self.normalization == "global":
            mean, std = self._compute_global_stats(splits["train"], dataset_root)
            self._set_global_stats(mean, std)
        
        self.train_ds = Sen12Landslides(
            files=splits["train"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.train_transforms,
        )
        self.val_ds = Sen12Landslides(
            files=splits["val"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.val_transforms,
        )
        self.test_ds = Sen12Landslides(
            files=splits["test"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.test_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )