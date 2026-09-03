# Adding a dataset

1. Record the exact dataset title, citation, official acquisition URL, source
   license, redistribution policy, coordinate system, timestamp semantics, and
   split protocol in a dataset config.
2. Keep raw files below `data/raw/`; they are ignored by Git and are never
   copied into the repository.
3. Implement `inspect_raw`, `prepare`, and `describe_license` on a loader. The
   loader should count malformed records by reason, normalize timestamps to UTC,
   preserve source IDs/users when permitted, and emit `TrajectoryInput` records.
4. Call `write_canonical_dataset` with an explicit projected CRS and version.
   Do not append undocumented columns to `points.npy`.
5. Generate standard and, where appropriate, user-held-out or temporal splits.
   Validate the output and retain the checksum manifest.
6. Add a small synthetic/raw sample fixture that contains no restricted data,
   plus focused tests for parsing, rejection counts, projection, and split
   leakage.

There is intentionally no generic download command in the foundation layer.
Acquisition instructions may be documented, but a source must be explicitly
approved before a downloader is added. In particular, `germany.yaml` is a
disabled gate because “Germany” does not identify one uniquely licensed
dataset.
