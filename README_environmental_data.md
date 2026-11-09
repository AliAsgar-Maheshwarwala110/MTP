# Environmental data pipeline

The pipeline uses the repository's verified `patch_locations.geojson` coordinates
(WGS84), split metadata, and NetCDF acquisition dates. NetCDF `x/y` and
`center_lat/center_lon` are projected coordinates and are not treated as
latitude/longitude. Event dates are daily dates; features therefore use strictly
pre-event calendar days (`date < event_date`) to avoid leakage.

## Quick test

From the repository root:

```bash
cp config.yaml config.test.yaml
# edit config.test.yaml: dataset.limit: 3
env/bin/python scripts/inspect_dataset.py --limit 3
env/bin/python scripts/download_environmental_data.py --config config.test.yaml
env/bin/python scripts/build_environmental_features.py --config config.test.yaml
```

NASA POWER is a documented public daily point API. Its returned temperature and
precipitation fields are retained with the API's daily units (temperature °C,
precipitation mm, relative humidity %). POWER represents a coarse grid cell; it
is never resized or described as 10 m data. Requests are cached and reruns are
idempotent. ERA5-Land, GPM IMERG and CHIRPS are configuration switches reserved
for their respective gridded/download adapters; enabling them currently records
`not_implemented` in the manifest rather than producing fabricated observations.
ERA5-Land additionally requires a CDS account and `cdsapi` credentials.

For aligned multimodal detection, `configs/environmental_e8.yaml` selects the
strict S1/S2-aligned S12LS-AD splits. POWER requests are cached using
0.1-degree representative coordinates, then expanded to all matching patch
IDs. This reduces redundant queries while preserving the source's coarse-grid
meaning; it does not create patch-resolution weather observations.

Outputs are written below the configured `output_dir`: daily observations,
one-row-per-patch features, event metadata, a download manifest, and a feature
dictionary. Join on `sample_id`; event identifiers, dates, and coordinates are
not model inputs. `scripts/run_e8_s1_s2_weather.sh` runs the full weather
download, feature build, and lightweight S1/S2/weather landslide-detection
training pipeline, with timestamped progress logs and measured ETA.
