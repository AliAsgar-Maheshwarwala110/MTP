"""Build deterministic event-relative features from daily_weather.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


WINDOWS = (1, 3, 7, 14, 30)


def aggregate_daily(daily: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    weather_columns = ("prectotcorr", "t2m", "t2m_min", "t2m_max", "rh2m")
    for column in weather_columns:
        if column in daily:
            daily[column] = pd.to_numeric(daily[column], errors="coerce").replace(-999, np.nan)
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily["event_date"] = pd.to_datetime(daily["event_date"])
    grouped = (
        {sample_id: group for sample_id, group in daily.groupby("sample_id", sort=False)}
        if "sample_id" in daily else {}
    )
    rows = []
    for sample in samples.itertuples(index=False):
        sample_id = sample.sample_id
        group = grouped.get(sample_id, daily.iloc[0:0])
        event = pd.to_datetime(sample.event_date)
        if not group.empty:
            group = group[group.event_date == event]
        row = {"sample_id": sample_id, "data_available": not group.empty}
        for days in WINDOWS:
            period = group[(group.date < event) & (group.date >= event - pd.Timedelta(days=days))]
            precipitation = period.get("prectotcorr", pd.Series(dtype=float))
            row[f"precipitation_{days}d_mm"] = precipitation.sum(min_count=1)
            if days in (7, 30):
                row[f"precipitation_max_daily_{days}d_mm"] = precipitation.max()
            if days == 7:
                for column, output in (
                    ("t2m", "temperature_mean_7d_c"),
                    ("t2m_min", "temperature_min_7d_c"),
                    ("t2m_max", "temperature_max_7d_c"),
                    ("rh2m", "relative_humidity_mean_7d_pct"),
                ):
                    row[output] = period[column].mean() if column in period else np.nan
            row[f"missing_precipitation_{days}d"] = int(
                precipitation.isna().any() or period.empty
            )
        rows.append(row)
    return samples.merge(pd.DataFrame(rows), on="sample_id", how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg.get("root_dir", args.config.parent)) / cfg["output_dir"]
    samples = pd.read_csv(out / "sample_metadata.csv")
    daily = pd.read_csv(out / "daily_weather.csv")
    features = aggregate_daily(daily, samples)
    features.to_csv(out / "sample_environmental_features.csv", index=False)
    event = samples.drop_duplicates("event_id").copy()
    event.to_csv(out / "event_metadata.csv", index=False)
    dictionary = pd.DataFrame([
        {"column": c, "definition": "Generated feature or source metadata", "source": "NASA POWER or dataset metadata",
         "unit": "see column name", "temporal_window": "pre-event", "aggregation": "see column name",
         "missing_convention": "NaN plus missing_* indicator"}
        for c in features.columns
    ])
    dictionary.to_csv(out / "feature_dictionary.csv", index=False)


if __name__ == "__main__":
    main()
