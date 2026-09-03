# Reproducibility

Preparation is deterministic when the raw input bytes, resolved config, code,
seed, and version are unchanged. Stable ID ordering uses the named
`sha256-id-v1` algorithm. User-held-out splits group complete users before
assigning partitions, and temporal splits sort by UTC start time.

Raw checksums and the preprocessing configuration hash are recorded in
`dataset.json`. Canonical files are hashed in `checksums.sha256`; a second
preparation of the same valid version is idempotent, while a conflicting
existing version asks for a new version rather than overwriting data.

The synthetic fixture is the unrestricted CPU smoke dataset. Porto and GeoLife
loaders accept only user-supplied raw files and never fetch or redistribute
mobility data. The Germany config remains disabled until a supervisor approves
an exact source, schema, license, and comparison protocol.

For an auditable run, save the resolved YAML (`dump_resolved_yaml`), its
`resolved_hash`, the dataset checksum manifest, hardware metadata, and the
loader inspection report. Volatile wall-clock timestamps are not used in the
configuration hash.
