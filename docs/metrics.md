# Task and diagnostic metrics

Task construction is independent of method evaluation. Every task artifact records its schema version, generator version, resolved configuration, seed, content hash, and construction quality report.

## Oracle retrieval

Distances use the benchmark convention that lower is more similar. For each query, self matches are excluded before ranking. Equal oracle distances retain tie groups, then sort candidates by stable candidate ID. Ranking agreement and retrieval metrics must consume the same fixed query/database ID sets; arbitrary distance scales are not compared unless a calibration policy is recorded.

## Triplet diagnostics

For a record `(query, A, B)`, strict accuracy is the fraction of specified records for which the method ranks the expected candidate first. `unspecified` records are excluded from the denominator and reported as coverage. Tie-aware accuracy counts a method tie as correct for an expected tie and does not treat a strict preference as a tie. A notion's `tie_tolerance` and `minimum_margin` are stored with the notion, never inferred from method outputs.

## Construction yield

Random and hard-negative generators report candidates examined, accepted, yield, rejection reasons, thresholds, and achieved constraint values. A minimum-yield/required-count gate is explicit. When a hard-negative search misses its gate, the artifact is incomplete/failed quality rather than silently replaced with random negatives.

## Route overlap

Free-space v1 uses `resampled_path_coverage_v1`: resample both projected polylines by arc length, measure the fraction of samples within a configured metric tolerance of the other path, and average the two directed coverages. This is not road-segment overlap and must not be labeled as map-matched overlap.

## Robustness-ready fields

Perturbation records include requested and realized severity, units, input/output hashes, and quality flags. A later robustness stage may summarize distance or retrieval change by severity, but it must retain the transformation, notion, oracle, sample count, and failed/not-generated count. Missing groups are missing—not zero.
