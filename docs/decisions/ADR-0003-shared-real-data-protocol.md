# ADR-0003: Shared Real-Data Protocol and Dataset Gates

- Status: accepted
- Date: 2026-09-05

## Decision

Use `docs/benchmark-v1-protocol.md` as the authoritative standard-benchmark
protocol.  Porto is the first eligible real-data candidate because its UCI
source records CC BY 4.0.  GeoLife and T-Drive remain disabled until the exact
download terms are recorded.  AIS remains disabled until a release, region,
time period, vessel-selection rules, projected CRS, and terms are fixed.

The free-space GPS track is the main benchmark.  Road-network work is a
separate gated track; it cannot modify the canonical free-space data or be
called equivalent to free-space evaluation.

## Consequences

No raw data is downloaded, redistributed, or evaluated merely because a
configuration exists.  The next executable gate is a Porto raw-file checksum
and canonical preparation report.  Paper reproduction and standard benchmark
results must remain separate in all tables and dashboard views.
