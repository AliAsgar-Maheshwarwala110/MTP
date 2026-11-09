"""Download cached daily environmental observations for dataset samples.

NASA POWER is a point API and is enabled by default. ERA5-Land, GPM IMERG and
CHIRPS are intentionally opt-in; their adapters record setup/API failures in
the manifest instead of silently substituting another source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

try:
    from scripts.inspect_dataset import inspect
except ModuleNotFoundError:
    from inspect_dataset import inspect

LOG = logging.getLogger("environmental")
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M"


def _sample_table(root: Path, cfg: dict) -> pd.DataFrame:
    if cfg["dataset"].get("aligned", False):
        base = root / "tasks" / cfg["dataset"]["task"] / cfg["dataset"]["version"]
        splits = json.loads((base / "splits_aligned.json").read_text())
        locations = json.loads((base / "patch_locations_aligned.geojson").read_text())
        location_index = {
            feature["properties"]["patch_id"]: feature["properties"]
            for feature in locations["features"]
        }
        rows = []
        for split, entries in splits.items():
            for entry in entries:
                location = location_index.get(entry["id"])
                if location is None:
                    raise ValueError(f"No aligned patch location for {entry['id']}")
                rows.append({
                    **entry,
                    "sample_id": entry["id"],
                    "split": split,
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "event_date": location["event_date"],
                    "event_id": location["inventory"],
                })
        frame = pd.DataFrame(rows)
        if frame["sample_id"].duplicated().any():
            raise ValueError("Aligned split contains duplicate patch IDs")
        return frame

    result = inspect(root, cfg["dataset"]["task"], cfg["dataset"]["version"],
                     cfg["dataset"]["modality"], cfg["dataset"].get("limit"))
    rows = result["samples"]
    frame = pd.DataFrame(rows)
    frame["event_date"] = pd.to_datetime(
        frame.get("geo_event_date", frame.get("split_event_date", frame.get("netcdf_event_date"))),
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    frame["event_id"] = frame.get("geo_inventory", frame.get("split_inventory", "")).fillna("").astype(str)
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")
    return frame


def _request_power(session: requests.Session, lat: float, lon: float, start: str, end: str,
                   timeout: int, retries: int) -> dict:
    params = {"latitude": lat, "longitude": lon, "start": start.replace("-", ""),
              "end": end.replace("-", ""), "community": "AG", "parameters": POWER_PARAMETERS,
              "format": "JSON"}
    last = None
    for attempt in range(retries):
        try:
            response = session.get(POWER_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "properties" not in payload or "parameter" not in payload["properties"]:
                raise ValueError("NASA POWER response lacks properties.parameter")
            return payload["properties"]["parameter"]
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"NASA POWER request failed after {retries} attempts: {last}") from last


def download(root: Path, cfg: dict) -> None:
    out = root / cfg["output_dir"]
    cache = out / "cache" / "power"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    samples = _sample_table(root, cfg)
    valid = samples.dropna(subset=["latitude", "longitude", "event_date"]).copy()
    if ((valid.latitude.abs() > 90) | (valid.longitude.abs() > 180)).any():
        raise ValueError("Invalid geographic coordinates found after GeoJSON matching")
    window = int(cfg["dataset"].get("days_before", 30))
    manifest = []
    daily = []
    session = requests.Session()
    enabled = cfg["sources"].get("nasa_power", {}).get("enabled", True)
    if enabled:
        groups = {}
        for _, sample in valid.iterrows():
            # POWER is a coarse-grid source; avoid redundant point queries for
            # patches sharing the same rounded representative coordinate.
            groups.setdefault(
                (round(sample.latitude, 1), round(sample.longitude, 1),
                 sample.event_date),
                [],
            ).append(sample)
        group_count = len(groups)
        started = time.monotonic()
        for group_index, ((latitude, longitude, event_date), group_samples) in enumerate(
            groups.items(), start=1
        ):
            status = "failed"
            end = event_date
            start = (pd.Timestamp(end) - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
            key = hashlib.sha256(
                f"{latitude:.1f},{longitude:.1f},{start},{end}".encode()
            ).hexdigest()[:24]
            target = cache / f"{key}.json"
            try:
                if target.exists():
                    parameters = json.loads(target.read_text())
                    status = "cached"
                else:
                    parameters = _request_power(
                        session, latitude, longitude, start, end,
                        int(cfg["http"]["timeout_seconds"]),
                        int(cfg["http"]["retries"]),
                    )
                    target.write_text(json.dumps(parameters))
                    status = "downloaded"
                for sample in group_samples:
                    record = {
                        "source": "nasa_power",
                        "sample_id": sample.sample_id,
                        "requested_start": start,
                        "requested_end": end,
                        "requested_latitude": latitude,
                        "requested_longitude": longitude,
                        "path": str(target),
                        "status": status,
                        "retrieved_at_utc": pd.Timestamp.now("UTC").isoformat(),
                    }
                    for day in sorted({d for p in parameters.values() for d in p}):
                        row = {
                            "sample_id": sample.sample_id,
                            "event_id": sample.event_id,
                            "latitude": sample.latitude,
                            "longitude": sample.longitude,
                            "requested_latitude": latitude,
                            "requested_longitude": longitude,
                            "date": pd.to_datetime(day).strftime("%Y-%m-%d"),
                            "source": "nasa_power",
                            "source_product": "NASA POWER Daily API v2",
                            "units": "T2M/T2M_MIN/T2M_MAX: °C; PRECTOTCORR: mm/day; RH2M: %",
                            "time_standard": "LST",
                            "matching_method": "NASA POWER point query at 0.1-degree representative; native ~0.5-degree grid",
                            "event_date": sample.event_date,
                        }
                        row.update({name.lower(): values.get(day) for name, values in parameters.items()})
                        daily.append(row)
                    manifest.append(record)
            except Exception as exc:
                for sample in group_samples:
                    LOG.error("sample %s: %s", sample.sample_id, exc)
                    manifest.append({
                        "source": "nasa_power",
                        "sample_id": sample.sample_id,
                        "requested_start": start,
                        "requested_end": end,
                        "requested_latitude": latitude,
                        "requested_longitude": longitude,
                        "path": str(target),
                        "status": "failed",
                        "error": str(exc),
                    })
            if group_index % 25 == 0 or group_index == group_count:
                elapsed = time.monotonic() - started
                remaining = elapsed / group_index * (group_count - group_index)
                completed_samples = sum(
                    len(v) for v in list(groups.values())[:group_index]
                )
                LOG.info(
                    "POWER groups %d/%d; samples %d/%d; elapsed %.1f min; ETA %.1f min",
                    group_index, group_count,
                    completed_samples,
                    len(valid), elapsed / 60, remaining / 60,
                )
            if status != "cached":
                time.sleep(float(cfg["http"].get("rate_limit_seconds", 0.2)))
    for source, settings in cfg["sources"].items():
        if source != "nasa_power" and settings.get("enabled"):
            manifest.append({"source": source, "status": "not_implemented",
                             "error": "Enable the documented adapter after configuring credentials/product access."})
    pd.DataFrame(daily).drop_duplicates(["sample_id", "date", "source"]).to_csv(out / "daily_weather.csv", index=False)
    pd.DataFrame(manifest).drop_duplicates(["source", "sample_id", "requested_start", "requested_end"],
                                           keep="last").to_csv(out / "download_manifest.csv", index=False)
    samples.to_csv(out / "sample_metadata.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(Path(cfg.get("root_dir", args.config.parent)), cfg)


if __name__ == "__main__":
    main()
