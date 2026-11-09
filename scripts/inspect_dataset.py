"""Inspect Sen12Landslides metadata without changing the downloaded dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xarray as xr


def _geojson_index(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {f["properties"]["patch_id"]: f["properties"] for f in data.get("features", [])}


def inspect(root: Path, task: str, version: str, modality: str, limit: int | None = None) -> dict:
    base = root / "tasks" / task / version
    split_path = base / modality / "splits.json"
    splits = json.loads(split_path.read_text())
    locations = _geojson_index(base / modality / "patch_locations.geojson")
    rows = []
    for split, entries in splits.items():
        for entry in entries[: limit or None]:
            patch_id = entry.get("id")
            nc = root / "data" / f"data_{version}" / entry[modality]
            row = {"sample_id": patch_id, "split": split, "file": str(nc)}
            row.update({f"split_{k}": v for k, v in entry.items() if k not in {"id", modality}})
            row.update({f"geo_{k}": v for k, v in locations.get(patch_id, {}).items()})
            if nc.exists():
                with xr.open_dataset(nc) as ds:
                    row.update({
                        "netcdf_variables": ",".join(ds.data_vars),
                        "netcdf_dimensions": json.dumps(dict(ds.sizes)),
                        "netcdf_crs": ds.attrs.get("crs", ""),
                        "netcdf_event_date": ds.attrs.get("event_date", ""),
                        "netcdf_time_start": str(ds.time.values[0])[:10] if "time" in ds else "",
                        "netcdf_time_end": str(ds.time.values[-1])[:10] if "time" in ds else "",
                        "satellite_acquisition_dates": json.dumps(
                            [str(value)[:10] for value in ds.time.values]
                        ) if "time" in ds else "[]",
                        "netcdf_pre_post_dates": str(ds.attrs.get("pre_post_dates", "")),
                    })
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and {"geo_lat", "geo_lon"}.issubset(frame):
        frame["latitude"] = pd.to_numeric(frame["geo_lat"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["geo_lon"], errors="coerce")
    return {
        "root": str(root),
        "task": task,
        "version": version,
        "modality": modality,
        "samples_inspected": len(frame),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "location_features": len(locations),
        "samples": frame.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task", default="S12LS-AD")
    parser.add_argument("--version", default="harmonized")
    parser.add_argument("--modality", default="s2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("environmental_data/dataset_inspection.json"))
    args = parser.parse_args()
    result = inspect(args.root, args.task, args.version, args.modality, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
