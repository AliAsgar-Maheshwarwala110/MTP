"""Aligned Sentinel-1/2 and event-relative NASA POWER patch dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.datasets import Sen12Landslides


WEATHER_COLUMNS = (
    "precipitation_1d_mm",
    "precipitation_3d_mm",
    "precipitation_7d_mm",
    "precipitation_max_daily_7d_mm",
    "precipitation_14d_mm",
    "precipitation_30d_mm",
    "precipitation_max_daily_30d_mm",
    "temperature_mean_7d_c",
    "temperature_min_7d_c",
    "temperature_max_7d_c",
    "relative_humidity_mean_7d_pct",
)
WEATHER_MISSING_COLUMNS = (
    "missing_precipitation_1d",
    "missing_precipitation_3d",
    "missing_precipitation_7d",
    "missing_precipitation_14d",
    "missing_precipitation_30d",
)
DAILY_COLUMNS = ("prectotcorr", "t2m", "t2m_min", "t2m_max", "rh2m")
SEQUENCE_LENGTH = 30


def _read_modality(entry: dict, modality: str) -> dict:
    return Sen12Landslides(
        files=[entry],
        modality=modality,
        min_date="2015-12-03",
        use_dem=True,
    )[0]


def compute_image_statistics(entries: list[dict], modality: str, max_samples: int) -> dict:
    source_entries = entries[:max_samples]
    total = None
    total_sq = None
    count = None
    for entry in source_entries:
        image = _read_modality(entry, modality)["img"].numpy().astype(np.float64)
        valid = np.isfinite(image) & (image > -9000)
        values = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        values = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
        valid = valid.transpose(1, 0, 2, 3).reshape(values.shape)
        if total is None:
            total = np.zeros(values.shape[0], dtype=np.float64)
            total_sq = np.zeros(values.shape[0], dtype=np.float64)
            count = np.zeros(values.shape[0], dtype=np.float64)
        total += np.where(valid, values, 0).sum(axis=1)
        total_sq += np.where(valid, values * values, 0).sum(axis=1)
        count += valid.sum(axis=1)
    if total is None or np.any(count == 0):
        raise RuntimeError(f"Unable to calculate {modality} normalization statistics")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "samples": len(source_entries)}


def prepare_weather_features(
    features: pd.DataFrame, train_ids: set[str]
) -> tuple[dict[str, np.ndarray], dict]:
    indexed = features.set_index("sample_id")
    missing_columns = [column for column in (*WEATHER_COLUMNS, *WEATHER_MISSING_COLUMNS)
                       if column not in indexed]
    if missing_columns:
        raise ValueError(f"Weather features are missing columns: {missing_columns}")
    training = indexed.loc[sorted(train_ids), list(WEATHER_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = training.median()
    if medians.isna().any():
        raise ValueError(
            "Cannot normalize weather features with no valid training observations: "
            f"{medians[medians.isna()].index.tolist()}"
        )
    training_filled = training.fillna(medians)
    means = training_filled.mean()
    stds = training_filled.std(ddof=0).replace(0, 1)

    arrays = {}
    for sample_id, row in indexed.iterrows():
        values = pd.to_numeric(row[list(WEATHER_COLUMNS)], errors="coerce")
        missing = values.isna().to_numpy(dtype=np.float32)
        normalized = ((values.fillna(medians) - means) / stds).clip(-5, 5)
        source_missing = pd.to_numeric(
            row[list(WEATHER_MISSING_COLUMNS)], errors="coerce"
        ).fillna(1).to_numpy(dtype=np.float32)
        arrays[sample_id] = np.concatenate(
            [normalized.to_numpy(dtype=np.float32), missing, source_missing]
        )
    stats = {
        "columns": list(WEATHER_COLUMNS),
        "missing_columns": list(WEATHER_MISSING_COLUMNS),
        "median": {key: float(value) for key, value in medians.to_dict().items()},
        "mean": {key: float(value) for key, value in means.to_dict().items()},
        "std": {key: float(value) for key, value in stds.to_dict().items()},
        "training_sample_count": len(training),
    }
    return arrays, stats


def prepare_weather_sequences(
    daily: pd.DataFrame,
    event_dates: dict[str, str],
    train_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict]:
    """Create standardized daily weather sequences using train-only statistics."""
    data = daily.copy()
    data["sample_id"] = data["sample_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in DAILY_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(-999, np.nan)
    data["prectotcorr"] = np.log1p(data["prectotcorr"].clip(lower=0))
    grouped = {sample_id: group.set_index("date") for sample_id, group in data.groupby("sample_id")}

    raw_sequences = {}
    for sample_id, event_date in event_dates.items():
        event = pd.Timestamp(event_date)
        dates = pd.date_range(end=event - pd.Timedelta(days=1), periods=SEQUENCE_LENGTH, freq="D")
        group = grouped.get(sample_id)
        if group is None:
            values = np.full((SEQUENCE_LENGTH, len(DAILY_COLUMNS)), np.nan, dtype=np.float32)
        else:
            values = group.reindex(dates)[list(DAILY_COLUMNS)].to_numpy(dtype=np.float32)
        raw_sequences[sample_id] = values

    training = np.stack([raw_sequences[sample_id] for sample_id in sorted(train_ids)])
    medians = np.nanmedian(training, axis=(0, 1))
    if not np.isfinite(medians).all():
        raise ValueError("No valid training observations for one or more daily weather channels")
    means = np.nanmean(training, axis=(0, 1))
    stds = np.nanstd(training, axis=(0, 1))
    stds[stds == 0] = 1

    sequences = {}
    time_axis = np.linspace(-1.0, 0.0, SEQUENCE_LENGTH, dtype=np.float32)[:, None]
    for sample_id, values in raw_sequences.items():
        missing = ~np.isfinite(values)
        filled = np.where(missing, medians[None, :], values)
        normalized = np.clip((filled - means[None, :]) / stds[None, :], -5, 5)
        sequences[sample_id] = np.concatenate(
            [normalized, missing.astype(np.float32), np.broadcast_to(time_axis, (SEQUENCE_LENGTH, 1))],
            axis=1,
        ).astype(np.float32)
    stats = {
        "columns": list(DAILY_COLUMNS),
        "sequence_length_days": SEQUENCE_LENGTH,
        "precipitation_transform": "log1p(max(mm/day, 0))",
        "median": {column: float(value) for column, value in zip(DAILY_COLUMNS, medians)},
        "mean": {column: float(value) for column, value in zip(DAILY_COLUMNS, means)},
        "std": {column: float(value) for column, value in zip(DAILY_COLUMNS, stds)},
        "training_sample_count": len(train_ids),
    }
    return sequences, stats


class S1S2WeatherDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        weather: dict[str, np.ndarray],
        weather_sequences: dict[str, np.ndarray],
        image_stats: dict[str, dict],
        ablation: str,
    ):
        self.entries = entries
        self.weather = weather
        self.weather_sequences = weather_sequences
        self.image_stats = image_stats
        self.ablation = ablation

    def __len__(self):
        return len(self.entries)

    def _normalize(self, image: torch.Tensor, modality: str) -> torch.Tensor:
        stats = self.image_stats[modality]
        mean = torch.tensor(stats["mean"], dtype=image.dtype).view(1, -1, 1, 1)
        std = torch.tensor(stats["std"], dtype=image.dtype).view(1, -1, 1, 1)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return ((image - mean) / (std + 1e-6)).clamp(-5, 5)

    def __getitem__(self, index):
        entry = self.entries[index]
        modalities = ["s2"]
        if self.ablation != "s2":
            modalities.extend(("s1asc", "s1dsc"))
        images = {modality: _read_modality(entry, modality) for modality in modalities}
        normalized = {
            modality: self._normalize(sample["img"].float(), modality).mean(dim=0)
            for modality, sample in images.items()
        }
        sample = {
            "s2": normalized["s2"],
            "label": (images["s2"]["msk"] > 0).any().float(),
        }
        if self.ablation != "s2":
            sample.update(s1asc=normalized["s1asc"], s1dsc=normalized["s1dsc"])
        if self.ablation == "s1_s2_weather":
            sample.update(
                weather=torch.as_tensor(self.weather[entry["id"]], dtype=torch.float32),
                weather_sequence=torch.from_numpy(self.weather_sequences[entry["id"]]),
            )
        return sample


def load_aligned_data(
    root: Path,
    weather_dir: Path,
    stats_samples: int = 256,
    ablation: str = "s1_s2_weather",
):
    if ablation not in ("s2", "s1_s2", "s1_s2_weather"):
        raise ValueError(f"Unknown ablation: {ablation}")
    task_dir = root / "tasks" / "S12LS-AD" / "harmonized"
    splits = json.loads((task_dir / "splits_aligned.json").read_text())
    data_root = root / "data" / "data_harmonized"
    for rows in splits.values():
        for entry in rows:
            for modality in ("s1asc", "s1dsc", "s2"):
                entry[modality] = str(data_root / entry[modality])

    weather, weather_sequences, weather_stats = {}, {}, {}
    if ablation == "s1_s2_weather":
        features = pd.read_csv(weather_dir / "sample_environmental_features.csv")
        expected_ids = {entry["id"] for rows in splits.values() for entry in rows}
        actual_ids = set(features["sample_id"].astype(str))
        if expected_ids != actual_ids:
            raise ValueError(
                "Weather feature coverage does not match aligned patches "
                f"(missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)})"
            )
        if features["sample_id"].astype(str).duplicated().any():
            raise ValueError("Weather feature table has duplicate sample IDs")
        if not features["data_available"].fillna(False).astype(bool).all():
            missing = features.loc[~features["data_available"].fillna(False), "sample_id"].tolist()
            raise ValueError(f"Weather data is unavailable for {len(missing)} aligned patches")
        train_ids = {entry["id"] for entry in splits["train"]}
        weather, static_stats = prepare_weather_features(features, train_ids)
        event_dates = dict(zip(features["sample_id"].astype(str), features["event_date"]))
        daily = pd.read_csv(weather_dir / "daily_weather.csv")
        weather_sequences, sequence_stats = prepare_weather_sequences(daily, event_dates, train_ids)
        weather_stats = {"aggregates": static_stats, "daily_sequence": sequence_stats}
    modalities = ["s2"]
    if ablation != "s2":
        modalities.extend(("s1asc", "s1dsc"))
    image_stats = {
        modality: compute_image_statistics(splits["train"], modality, stats_samples)
        for modality in modalities
    }
    return {
        split: S1S2WeatherDataset(entries, weather, weather_sequences, image_stats, ablation)
        for split, entries in splits.items()
    }, image_stats, weather_stats, splits
