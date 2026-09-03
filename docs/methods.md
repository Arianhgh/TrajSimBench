# Classical trajectory measures (v1)

This document freezes the Phase 2 measure semantics. Every implementation is
CPU-only and uses NumPy `float64` projected coordinates. The canonical
foundation point layout is `[lon_deg, lat_deg, x_m, y_m, timestamp_s]`; small
standalone examples may pass `[x, y]`. If a foundation view provides a
`metadata["projected_points"]` array, that array is used instead.

All public methods implement the typed `TrajectoryMeasure` contract:

```python
from trajsimbench.measures import TrajectoryView, registry

query = TrajectoryView("q", query_points, {})
candidate = TrajectoryView("c", candidate_points, {})
for measure in registry:
    result = measure.distance(query, candidate)
    print(measure.name, result.distance)
```

`DistanceResult.distance` is finite, non-negative, and lower-is-more-similar.
`raw_score` retains the unnormalized cumulative/edit/LCSS score. `runtime_ns`
is measured by the public wrapper, and `details` contains method-specific
provenance. Classical methods are deterministic and symmetric as implemented.

## Shared input and edge-case policy

- Empty trajectories are rejected with `ValueError`; there is no mathematically
  useful canonical normalization for a zero-length input in this benchmark.
- Inputs must be two-dimensional and contain at least two coordinates. Projected
  coordinates must be finite. A timestamp column may contain `NaN` only when
  timestamps are genuinely unavailable; it is never used as a spatial feature.
- One-point paths are valid. Their polyline length is zero, and measures use
  their ordinary one-point recurrences.
- Repeated points are valid. Arc-length resampling collapses repeated
  cumulative positions without changing the geometric path.
- Unequal point counts are valid for every measure. Only an explicitly narrow
  DTW window can make a pair infeasible; that case raises a clear `ValueError`.
- Identical trajectories have distance zero for all seven methods (provided
  LCSS/EDR epsilon permits exact matches, as it does at the default threshold).
  Reversing a non-palindromic path is not treated as identical by sequence-aware
  methods; Hausdorff intentionally ignores order.
- NaN or infinite projected coordinates are rejected. NaN timestamps are
  treated as missing; requesting an LCSS `time_delta_s` then raises unless both
  paths have finite aligned timestamps.
- Epsilon accepts zero, ordinary non-negative values, and positive infinity.
  Zero means exact projected-coordinate matching; infinity matches every pair
  of finite points for LCSS/EDR.

The implementation does not silently catch `NotImplementedError` to discover
capabilities. Callers inspect `measure.capabilities` and receive
`MeasureCapabilityError` for unsupported encoding/index operations.

## Configurations

Configurations are strict `BaseMethodConfig` models. Unknown keys and invalid
types/ranges are rejected. They support `model_validate`, `model_dump`, and
`dict` so the later YAML loader can use the same boundary with or without the
optional Pydantic dependency.

| Method | Config fields and defaults |
| --- | --- |
| `euclidean` | `n_samples=100` (aliases: `sampling_count`, `resample_count`, `num_samples`) |
| `dtw` | `normalization="none"`; `window=None` (aliases: `global_normalization`, `window_size`, `sakoe_chiba_window`) |
| `hausdorff` | no fields |
| `discrete_frechet` | no fields |
| `lcss` | `epsilon=1.0`, `delta_mode="index"`, `delta=None`, `time_delta_s=None` |
| `edr` | `epsilon=1.0` |
| `erp` | `gap_point=(0.0, 0.0)`, `normalization="none"`; `normalize=True` is an alias for `normalization="max_input_length"` |

The registry validates a mapping against the method-specific config model
before constructing a measure. Its metadata records the stable name, semantic
version, implementation, capabilities, config model, source, and citation
field for manifests.

## Definitions

### Resampled pointwise Euclidean (`euclidean`)

Let `P=(p_0,...,p_{n-1})` be a projected polyline. Compute cumulative arc
lengths `s_i`, normalize the interval to `[0,L]`, and linearly interpolate each
coordinate at the same `k=n_samples` equally spaced arc-length positions for
both paths. The distance is the mean pointwise cost:

\[
d_E(P,Q) = \frac{1}{k}\sum_{r=0}^{k-1}\|\tilde p_r-\tilde q_r\|_2.
\]

This is explicitly not index truncation or padding. A zero-length path is
represented by its repeated first point during resampling. The raw score is
the mean above.

### Dynamic time warping (`dtw`)

For projected point cost `c(i,j)=||p_i-q_j||_2`, initialize
`D[0,0]=0`, borders to infinity, and use

\[
D[i,j]=c(i,j)+\min(D[i-1,j],D[i,j-1],D[i-1,j-1]).
\]

`window` is an inclusive Sakoe--Chiba constraint `|i-j| <= window`. `None`
uses the full grid. The v1 oracle default is raw cumulative DTW (`none`).
`path_length` divides by the number of visited cells in a deterministically
backtraced optimal warping path (including both endpoints); ties prefer
diagonal, then up, then left. `max_input_length` divides by
`max(len(P),len(Q),1)`. `raw_score` is always the unnormalized `D[n,m]`. A
window narrower than the endpoint length gap is rejected rather than returning
infinity.

### Symmetric Hausdorff (`hausdorff`)

Order and time are ignored. For point sets `P,Q`,

\[
h(P,Q)=\max\{\max_{p\in P}\min_{q\in Q}\|p-q\|_2,
              \max_{q\in Q}\min_{p\in P}\|q-p\|_2\}.
\]

The two directed values are retained in `details`.

### Discrete Fréchet (`discrete_frechet`)

The recurrence is the standard monotone coupling recurrence with
`C[0,0]=c(0,0)`, border values formed by running maxima, and

\[
C[i,j]=\max\left(c(i,j),\min(C[i-1,j],C[i-1,j-1],C[i,j-1])\right).
\]

This preserves vertex order and is different from point-set Hausdorff.

### Longest common subsequence (`lcss`)

Two points match when their projected distance is at most `epsilon`, their
zero-based index separation is at most `delta` when `delta_mode="index"`, and
their absolute timestamp separation is at most `time_delta_s` when configured.
With `delta_mode="time"`, `delta` is interpreted as seconds. The
timestamp constraint requires finite timestamps on both inputs. With the
usual subsequence recurrence, let `\ell` be the maximum match count. The
canonical distance is

\[
d_{LCSS}(P,Q)=1-\frac{\ell}{\min(n,m)}.
\]

The `raw_score` is `\ell`. This normalization is bounded in `[0,1]` and is
defined for all non-empty inputs.

### Edit distance on real sequences (`edr`)

Substitution costs zero if projected point distance is at most `epsilon`, and
one otherwise. Insertions and deletions each cost one. The standard edit
recurrence is used, with edit cost `e` normalized as

\[
d_{EDR}(P,Q)=\frac{e}{\max(n,m)}.
\]

`raw_score` is the integer edit cost. This also places the canonical distance
in `[0,1]`.

### Edit distance with real penalty (`erp`)

ERP uses projected-vector Euclidean substitution cost and a configured finite
gap point `g`. Deleting `p_i` costs `||p_i-g||_2`, inserting `q_j` costs
`||q_j-g||_2`, and substituting costs `||p_i-q_j||_2`. The recurrence is

\[
E[i,j]=\min\begin{cases}
E[i-1,j-1]+\|p_i-q_j\|_2,\\
E[i-1,j]+\|p_i-g\|_2,\\
E[i,j-1]+\|q_j-g\|_2.
\end{cases}
\]

The raw cumulative score is the default canonical distance. Setting
`normalization="max_input_length"` (or `normalize=True`) divides it by
`max(n,m,1)`. The explicit v1 standalone default gap is `(0,0)` and is saved
in every resolved config; dataset-specific production configs should replace
it with a documented dataset gap vector rather than deriving a hidden
pair-specific gap.

## Correctness and performance

The tests under `tests/unit/test_measures*.py` cover hand cases and edge
conditions. `tests/reference/test_measures*.py` contains independent slow
reference recurrences, and `tests/property/test_measures*.py` exercises finite
outputs, identities, non-negativity, and symmetry over deterministic generated
paths. The optional `time_pairwise` helper is a warmed, single-process CPU
smoke harness only; it is not the formal Phase 14 systems benchmark.
