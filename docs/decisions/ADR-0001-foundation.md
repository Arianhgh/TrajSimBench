# ADR-0001: Foundation contracts

- Status: accepted
- Date: 2026-09-03

## Decision

Use strict Pydantic v2 configuration models, Pandas/Parquet metadata, NumPy
`float64` canonical point arrays, explicit projected CRS transformations, and
SHA-256 manifests. Expose read-only memory-mapped trajectory views. Keep raw
Porto and GeoLife data user-supplied and leave Germany disabled until its exact
source and license are approved.

## Rationale

These choices make misspelled research settings visible, prevent degree-based
distance mistakes, enable CPU-scale processing, and preserve provenance without
redistributing restricted mobility data. An atomic writer and immutable reader
also make accidental mutation and partial preparations detectable.

## Consequences

Callers must choose a CRS and dataset version explicitly. Optional method
parameters live under a method's `config` mapping, while experiment-level keys
are closed-world and reject typos. Future schema changes need a versioned
migration rather than silently accepting incompatible data.
