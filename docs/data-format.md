# Canonical data format

Each processed version lives at
`data/processed/<dataset>/<version>/` and contains:

```text
points.npy                 # C-contiguous float64 [N, 5]
offsets.npy                # int64 [trajectory_count + 1]
metadata.parquet           # one row per trajectory
dataset.json               # schema, source, CRS, counts, provenance
splits/<name>/*.npy        # ID arrays for train/val/test/query/database
checksums.sha256           # SHA-256 for every file above except itself
```

Optional point features are separate `feature_<name>.npy` files. Their paths,
dtype, and shape are declared in `dataset.json` and the first dimension must
equal the row count of `points.npy`.

The point columns are `lon_deg`, `lat_deg`, `x_m`, `y_m`, and `timestamp_s`.
The timestamp is Unix seconds in UTC or `NaN` when the source genuinely does
not provide one. `offsets[i]:offsets[i+1]` is the non-owning slice for trajectory
`i`; the reader marks it read-only so a perturbation cannot mutate the source.

Metadata includes stable namespaced IDs, optional pseudonymous users and source
IDs, projected CRS, point count, projected length, duration, split convenience
field, and JSON-encoded quality flags. Split files remain authoritative.

## CRS policy

Coordinates are preserved in WGS84 for display and export. All metric work uses
the explicitly resolved projected CRS, transformed with `always_xy=True`.
The writer recomputes projected columns from longitude/latitude; the validator
checks sampled points against the declared CRS.

## Validation

`validate_dataset(path)` checks file readability, offsets, metadata alignment,
IDs, finite and bounded coordinates, timestamps, lengths/durations, projection
consistency, split leakage, manifest counts, and checksums. It returns a
`ValidationReport`; `report.raise_if_invalid()` raises one actionable error
containing every detected failure.
