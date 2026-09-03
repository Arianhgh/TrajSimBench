# Implementation progress

TrajSimBench is implemented through the CPU-first reproducible MVP described
in `IMPLEMENTATION_PLAN.md`. The checklist below records the delivered scope
and the intentionally gated scope so a later expansion cannot silently change
the benchmark contract.

## Delivered

- [x] Strict versioned YAML configuration, stable resolved hashes, seed and
  resource policies, and CLI validation/resolution.
- [x] Canonical float64 trajectory storage with offsets, metadata, projected
  coordinates, checksums, manifests, splits, validation, and memory-mapped
  read-only access.
- [x] Synthetic fixtures plus user-supplied Porto and GeoLife loader contracts;
  preprocessing is deterministic and atomic.
- [x] Seven classical measures: resampled Euclidean, DTW, symmetric Hausdorff,
  discrete Fréchet, LCSS, EDR, and ERP, with a common typed interface and
  registry.
- [x] Seeded free-space perturbations, provenance, equivalence/oracle tasks,
  diagnostics, negative generation, and leakage-safe task helpers.
- [x] Exact chunked top-k retrieval, ranking metrics, agreement, robustness,
  statistics, fingerprint schemas, and failure accounting.
- [x] Parquet artifacts, schema validation, manifests, DuckDB views, staged
  resumable CPU runner, cache keys, and Tiny end-to-end execution.
- [x] Analysis table/figure specifications with provenance sidecars and an
  import-safe eight-page dashboard surface.
- [x] External learned-method adapter protocol and deterministic fake adapter
  environment; no model training is required for the CPU smoke suite.

## Explicit gates

- [ ] Germany remains disabled until an exact source, version, license, and
  redistribution decision is supplied.
- [ ] Named learned baselines remain protocol-only until their source code,
  weights, preprocessing, and license are approved.
- [ ] ANN and large systems tracks remain opt-in and are not run by the Tiny
  CI configuration.
- [ ] Cross-domain and application-specific claims require approved datasets
  and task annotations; the implementation does not fabricate them.

## Verification snapshot

The local CPU verification suite passes:

```text
pytest -q
ruff check .
ruff format --check .
mypy trajsimbench
```

The reproducible Tiny run is stored below `results/tiny_synthetic_cpu/` and
is validated by `trajbench results validate`; generated result directories are
ignored by Git so experiments do not pollute source changes.
