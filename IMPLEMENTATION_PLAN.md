# TrajSimBench Comprehensive Implementation Plan

## 0. How the implementing agent must use this plan

This repository is empty at the time this plan was written. Treat TrajSimBench as a greenfield Python project.

Implement the phases in order. Do not build the dashboard, integrate learned baselines, or optimize classical measures until the core data contracts, correctness tests, and Tiny benchmark are working. At the end of each phase:

1. run the phase's required checks;
2. fix failures before continuing;
3. update the checklist in this document or a dedicated progress file;
4. record any deliberate deviation in `docs/decisions/` as a short architecture decision record (ADR);
5. never silently invent a dataset source, license, similarity definition, threshold, or ground-truth label.

Commands shown in the research proposal are desired product interfaces, not instructions to execute while reading the proposal. The scientific claims in the proposal are hypotheses to test, not expected results to hard-code.

## 1. Product outcome

Build an open, reproducible trajectory-similarity benchmark that characterizes what each method considers similar. The completed system must support:

- canonical loading and preprocessing of trajectory datasets;
- at least six correct classical measures;
- five separate benchmark tracks: oracle approximation, underlying-trajectory equivalence, counterfactual diagnostics, application-specific evaluation, and systems evaluation;
- deterministic perturbations and counterfactual triplets with provenance;
- random and structured hard-negative retrieval;
- ranking agreement, retrieval, robustness, monotonicity, statistical, and systems metrics;
- configuration-driven batch execution with resumable artifacts and manifests;
- Parquet result storage queryable through DuckDB;
- isolated adapters for learned or representation-based baselines;
- an eight-page Streamlit research dashboard that consumes saved artifacts;
- a CPU-only Tiny benchmark suitable for CI;
- reproducible paper tables and figures generated from saved results.

The benchmark must report results per similarity notion and per oracle. It must not collapse them into one universal leaderboard.

## 2. Definition of done

### 2.1 Minimum viable research benchmark

The MVP is done when all of the following are true:

- Porto and GeoLife can be prepared reproducibly from user-supplied raw data.
- The Germany adapter is either implemented against an explicitly approved source or clearly reported as unavailable; no source may be guessed.
- The canonical on-disk representation and validation command work.
- Euclidean pointwise, DTW, symmetric Hausdorff, discrete Frechet, LCSS, EDR, and ERP implementations pass unit and reference tests.
- Oracle retrieval, equivalence retrieval, diagnostic triplets, realistic robustness, hard-negative retrieval, and systems timing run from YAML.
- Exact classical Top-K retrieval works at Tiny and Standard tiers.
- A complete CPU-only Tiny experiment produces all required result files and a complete manifest.
- Agreement matrices, similarity fingerprints, robustness curves, hard-negative comparisons, and core tables can be regenerated solely from saved results.
- The dashboard reads saved results and provides at least the Dataset, Pair, Counterfactual, Retrieval Disagreement, Robustness, and Fingerprint pages.
- A clean checkout can install, run tests, run the Tiny benchmark, regenerate its analysis, and launch the dashboard using documented commands.

### 2.2 Publication-oriented target

After the MVP, add:

- an approved third dataset;
- at least two learned or representation-based methods through isolated adapters;
- user-held-out, temporal hold-out, and compatible cross-dataset experiments;
- exact vector retrieval with FAISS FlatL2, then optional ANN evaluation;
- the full Accuracy-Efficiency and Experiment Builder dashboard pages;
- three training seeds where computationally feasible;
- complete confidence intervals, corrected pairwise comparisons, systems profiling, Pareto analysis, and break-even analysis.

### 2.3 Explicit non-goals for v1

- Do not create a new neural architecture.
- Do not claim a universally best similarity method.
- Do not redistribute restricted raw mobility data.
- Do not begin with road-network map matching or road-network detours.
- Do not install incompatible research repositories into the core environment.
- Do not add HNSW before FlatL2 has established the exact embedding-retrieval baseline.
- Do not run large experiments inside Streamlit.
- Do not build React/FastAPI unless a measured Streamlit limitation creates a concrete requirement.
- Do not implement human-subject annotation without ethics and supervision approval.

## 3. Decisions that remove ambiguity from the proposal

| Area | v1 decision | Reason |
| --- | --- | --- |
| Package and CLI names | Distribution/import package: `trajsimbench`; console command: `trajbench`. | Preserves the proposal's CLI examples without creating an invalid mixed package name. |
| Language | Python 3.11 and 3.12 supported; CI starts with 3.11. | Matches the proposal while limiting compatibility work. |
| Dataframes | Use Pandas in the core implementation. Accept Arrow tables at boundaries. | Most accessible implementation path and broad library compatibility. |
| Configuration | YAML parsed into strict Pydantic models; reject unknown keys. | Prevents misspelled research settings from silently changing experiments. |
| Coordinate policy | Preserve WGS84 longitude/latitude, but run all metric-distance algorithms on projected meters. | Prevents degree-based distance mistakes while retaining original coordinates for maps. |
| Score direction | The core API exposes a distance where lower always means more similar. Adapters convert external similarities to distances and preserve the raw score separately. | Makes ranking and triplet logic consistent. |
| Floating-point type | Store canonical point arrays as `float64`; learned adapters may export `float32` embeddings. | Classical distance and projection correctness take priority over small storage savings. |
| Result store | Partitioned Parquet is authoritative; DuckDB views are a query layer, not a second source of truth. | Avoids synchronizing two stores. |
| Dashboard | Streamlit multipage app using shared read-only service functions. | Matches the proposal and keeps benchmark logic out of the UI. |
| Classical optimization | Correct NumPy/SciPy implementation first; Numba is an optional extra after reference parity. | Separates correctness from performance. |
| Learned environments | One container/environment per external baseline using a file-level protocol. | Protects the core environment from dependency conflicts. |
| Experiment identity | `run_id` is a UUID; `experiment_id` is a human-readable config identifier; an artifact fingerprint hashes resolved config, code, data, and method versions. | Allows repeated seeds/runs without collisions and enables cache validation. |
| Diagnostic ambiguity | Expectations are notion-specific and may be `a_closer`, `b_closer`, `tie`, or `unspecified`; unspecified cases do not enter triplet accuracy. | The proposal contains “depends” and “partially preserve” cells that cannot be forced into binary labels. |
| Germany dataset | Create only a disabled configuration stub until the exact dataset name, source URL, schema, and license are approved. | “Germany” is not a uniquely identifiable dataset. |
| Application track | Define the extension interfaces in v1, but do not report application results without externally defined labels and a versioned relevance policy. | The proposal does not yet specify an application ground truth. |

## 4. Required repository layout

Create this structure. Keep modules small and keep dataset-, method-, and task-specific logic behind registries.

```text
TrajSimBench/
|-- pyproject.toml
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- CHANGELOG.md
|-- .gitignore
|-- .pre-commit-config.yaml
|-- configs/
|   |-- ci/tiny_synthetic.yaml
|   |-- datasets/{synthetic,porto,geolife,germany}.yaml
|   |-- methods/{euclidean,dtw,hausdorff,discrete_frechet,lcss,edr,erp}.yaml
|   |-- notions/v1.yaml
|   |-- scales/v1.yaml
|   `-- experiments/{core_porto,core_geolife,agreement,robustness,hard_negatives}.yaml
|-- data/
|   |-- raw/.gitkeep
|   |-- interim/.gitkeep
|   `-- processed/.gitkeep
|-- docs/
|   |-- architecture.md
|   |-- data-format.md
|   |-- methods.md
|   |-- metrics.md
|   |-- reproducibility.md
|   |-- adding-a-dataset.md
|   |-- adding-a-method.md
|   |-- baseline-adapter-protocol.md
|   `-- decisions/
|-- trajsimbench/
|   |-- __init__.py
|   |-- cli.py
|   |-- config/
|   |   |-- models.py
|   |   |-- loader.py
|   |   `-- validation.py
|   |-- data/
|   |   |-- dataset.py
|   |   |-- schema.py
|   |   |-- validation.py
|   |   |-- projection.py
|   |   |-- splitting.py
|   |   |-- checksums.py
|   |   |-- loaders/{base,synthetic,porto,geolife,germany}.py
|   |   `-- preprocessing/{cleaning,resampling,statistics}.py
|   |-- measures/
|   |   |-- base.py
|   |   |-- registry.py
|   |   |-- result.py
|   |   |-- classical/{euclidean,dtw,hausdorff,discrete_frechet,lcss,edr,erp}.py
|   |   |-- learned/external.py
|   |   `-- lower_bound/
|   |-- perturbations/
|   |   |-- base.py
|   |   |-- registry.py
|   |   |-- result.py
|   |   |-- spatial.py
|   |   |-- temporal.py
|   |   |-- sampling.py
|   |   `-- route.py
|   |-- notions/{models,registry}.py
|   |-- tasks/{base,oracle,equivalence,diagnostics,retrieval,generalization,systems}.py
|   |-- negatives/{base,random,same_od,nearby_shape,translated,reversed,partial_overlap,temporal}.py
|   |-- retrieval/{exact,ranking,relevance}.py
|   |-- indexing/{base,numpy_flat,faiss_flat,faiss_hnsw}.py
|   |-- evaluation/{retrieval,agreement,diagnostics,robustness,statistics,systems,fingerprints}.py
|   |-- orchestration/{runner,stages,cache,resume,context}.py
|   |-- storage/{schemas,parquet,duckdb,manifest,artifacts}.py
|   |-- analysis/{tables,figures,pareto,break_even}.py
|   |-- dashboard/
|   |   |-- app.py
|   |   |-- services/{datasets,results,interactive}.py
|   |   |-- components/{maps,filters,tables}.py
|   |   `-- pages/{dataset,pair,counterfactual,disagreement,fingerprints,robustness,efficiency,builder}.py
|   `-- utils/{logging,seeding,hardware,timing,paths}.py
|-- baseline_envs/
|   |-- README.md
|   `-- <method>/{Dockerfile,runner.py,adapter.yaml,requirements.lock}
|-- scripts/{download,prepare,run_paper,generate_figures}/
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- reference/
|   |-- property/
|   |-- fixtures/
|   `-- data/toy/
|-- results/.gitkeep
`-- tools/
```

Do not create placeholder modules that merely contain `pass`. Add a module when its phase is implemented; use README stubs for future optional areas.

## 5. Dependency and packaging plan

Define a PEP 621 `pyproject.toml` and expose `trajbench = "trajsimbench.cli:app"`.

Core dependencies:

- NumPy, Pandas, PyArrow, DuckDB;
- SciPy and scikit-learn;
- Pydantic and PyYAML;
- PyProj and Shapely;
- Typer and Rich;
- psutil;
- platformdirs or an equivalent explicit cache-directory helper.

Optional dependency groups:

- `geo`: GeoPandas;
- `speed`: Numba;
- `index`: FAISS CPU, installed only on supported platforms;
- `dashboard`: Streamlit, Plotly, PyDeck;
- `learned`: PyTorch only for core-compatible adapters, not third-party repositories;
- `dev`: pytest, pytest-cov, Hypothesis, Ruff, mypy, pre-commit;
- `docs`: MkDocs and a minimal theme if documentation publishing is required.

Use a reproducible lock file supported by the chosen package manager. Avoid an unconditional FAISS or GPU dependency so the base package and Tiny CI remain portable.

Quality gates:

```powershell
ruff check .
ruff format --check .
mypy trajsimbench
pytest -q
```

Set an initial coverage floor of 85% for core modules. Exclude dashboard rendering and external baseline runners from the numeric gate, but integration-test their contracts.

## 6. Canonical data contract

### 6.1 Processed dataset directory

Each processed dataset version lives at `data/processed/<dataset>/<version>/`:

```text
points.npy
offsets.npy
metadata.parquet
dataset.json
splits/<split_name>/{train,val,test,query,database}.npy
checksums.sha256
```

`points.npy` is a C-contiguous `float64` array with columns:

1. `lon_deg`;
2. `lat_deg`;
3. `x_m`;
4. `y_m`;
5. `timestamp_s` as Unix seconds, or `NaN` if genuinely unavailable.

Put optional point features in separately named, row-aligned arrays described in `dataset.json`; do not append undocumented columns to `points.npy`.

`offsets.npy` is `int64`, has length `trajectory_count + 1`, begins at zero, is monotonically nondecreasing, and ends at `len(points)`. Trajectory `i` is `points[offsets[i]:offsets[i+1]]`.

`metadata.parquet` has one row per trajectory with this minimum schema:

| Field | Type | Rule |
| --- | --- | --- |
| `trajectory_idx` | int64 | Contiguous row index into offsets. |
| `trajectory_id` | string | Stable, namespaced, unique identifier. |
| `dataset` | string | Canonical dataset name. |
| `source_id` | string/null | Original source identifier if available. |
| `user_id` | string/null | Pseudonymous source user ID when permitted. |
| `start_time_s`, `end_time_s` | float64/null | UTC Unix time. |
| `mobility_mode` | string/null | Source label only; never inferred silently. |
| `length_m` | float64 | Sum of projected segment lengths. |
| `duration_s` | float64/null | Nonnegative. |
| `num_points` | int32 | Must equal the offset span. |
| `split` | string/null | Convenience field for one canonical split; split files remain authoritative. |
| `crs_projected` | string | EPSG or full CRS identifier used for `x_m`,`y_m`. |
| `quality_flags` | list<string> or JSON string | Cleaning/validation flags. |

`dataset.json` records schema version, source name, source URL, acquisition date, source license, redistribution policy, raw checksums, preprocessing config hash, code version, projected CRS policy, point feature declarations, counts, bounding box, and creation time.

### 6.2 In-memory API

Implement:

```python
@dataclass(frozen=True, slots=True)
class TrajectoryView:
    trajectory_id: str
    points: np.ndarray       # non-owning [n, 5] view
    metadata: Mapping[str, Any]

class TrajectoryDataset:
    @classmethod
    def open(cls, path: Path, *, mmap: bool = True) -> "TrajectoryDataset": ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> TrajectoryView: ...
    def by_id(self, trajectory_id: str) -> TrajectoryView: ...
    def ids(self, split: str | None = None) -> np.ndarray: ...
```

Default to memory mapping. Returned point views must be read-only so perturbations cannot mutate canonical data.

### 6.3 Validation invariants

The `trajbench validate-data` command must fail with actionable messages for:

- invalid offsets or row counts;
- duplicate trajectory IDs;
- trajectories below the configured minimum point count;
- non-finite spatial values;
- latitude/longitude outside valid ranges;
- non-monotonic timestamps, except when a dataset policy explicitly repairs or drops them;
- projected coordinates inconsistent with the declared CRS beyond a small tolerance on sampled points;
- negative duration/length;
- overlapping user-held-out users;
- missing split IDs or IDs in multiple mutually exclusive partitions;
- checksum mismatches.

### 6.4 Projection policy

Select a projected CRS through dataset configuration. For city-scale data, use an appropriate local UTM zone when all points fit safely. If a dataset crosses incompatible zones, define a documented local projection or geodesic preprocessing policy. Store the resolved CRS, never only the name of the selection heuristic. Transform with `always_xy=True`. All perturbation severities expressed in meters operate only on projected coordinates, after which longitude/latitude are inverse-transformed for export and display.

## 7. Dataset preparation

### 7.1 Loader interface

Each loader implements `inspect_raw`, `prepare`, and `describe_license`. Preparation must stream or chunk large raw files, normalize timestamps to UTC, create stable IDs, project coordinates, calculate metadata, write to a temporary version directory, validate it, then atomically move it into the final processed path.

Never overwrite an existing processed version with different content. Require a new version or explicit recoverable deletion.

### 7.2 Synthetic fixture dataset

Implement first. Include straight lines, translated copies, reversed paths, variable sampling, small and large detours, crossing paths, identical endpoints with different routes, temporal warps, repeated points, and very short trajectories. This dataset powers unit tests, screenshots, and `configs/ci/tiny_synthetic.yaml` without requiring licensed data.

### 7.3 Porto

The config must declare raw CSV path, polyline field, timestamp semantics, missing-data sentinel handling, bounding box, minimum/maximum point counts, and projection. Preserve original trip ID if present. Reject malformed polylines and out-of-bounds points with counted reason codes. The download helper may provide acquisition instructions or retrieve from an officially permitted source, but raw data must remain ignored by Git.

### 7.4 GeoLife

Parse per-user trajectory files deterministically. Preserve source user grouping, timestamps, and transportation labels only where the source files provide them. Deduplicate repeated points/times by an explicit policy. Create both a standard deterministic split and a user-held-out split with disjoint user sets.

### 7.5 Germany gate

Before implementation, obtain and record:

- exact dataset title and citation;
- official acquisition location;
- raw format and coordinate system;
- license and redistribution constraints;
- intended comparison protocol.

Until these are approved, `configs/datasets/germany.yaml` must be marked `enabled: false` with `status: requires_source_decision`, and the CLI must explain the gate rather than fail with an import error.

### 7.6 Splits and scale tiers

Use stable ID-based selection with a named algorithm version and seed. Support:

- standard 70/10/20 train/validation/test;
- GeoLife user-held-out split;
- temporal earliest-70/next-10/latest-20 split where timestamps permit;
- cross-dataset train/test references;
- Tiny: database 1,000, queries 100;
- Standard: database 10,000, queries 1,000;
- Medium: database 50,000, queries 200-500;
- Large: database 100,000+, queries 100-200.

When a dataset is too small, fail or use an explicitly configured reduced tier; never silently sample with replacement. Exclude the query itself from its candidate database by ID.

## 8. Similarity measure contract

### 8.1 Core interface

Use a typed capability-based interface, not feature detection by catching `NotImplementedError`:

```python
@dataclass(frozen=True)
class MeasureCapabilities:
    learned: bool = False
    supports_batch: bool = False
    supports_encoding: bool = False
    supports_index: bool = False
    symmetric: bool = True
    requires_timestamps: bool = False

@dataclass(frozen=True)
class DistanceResult:
    distance: float
    raw_score: float
    runtime_ns: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

class TrajectoryMeasure(ABC):
    name: str
    version: str
    capabilities: MeasureCapabilities
    config: BaseModel

    def fit(self, train_set, val_set=None) -> "TrajectoryMeasure": ...
    def encode(self, trajectories) -> np.ndarray: ...
    @abstractmethod
    def distance(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult: ...
    def pairwise(self, query, candidates) -> np.ndarray: ...
    def build_index(self, ids, embeddings=None): ...
    def top_k(self, query, k: int): ...
```

The canonical `distance` must be finite, deterministic for deterministic methods, and lower-is-more-similar. If an external method returns similarity, preserve it as `raw_score` and apply a documented monotonic conversion.

The registry maps stable names to factories and validates method-specific config before construction. Record implementation name, semantic version, config, dependency versions, and source/citation metadata in every run.

### 8.2 Classical definitions to freeze in `docs/methods.md`

Do not implement until each formula, parameter, and edge case below is documented and unit-tested.

- `euclidean`: resample each path to a configurable common count along normalized cumulative arc length, then compute mean pointwise Euclidean distance. This is not raw index truncation.
- `dtw`: dynamic programming over projected 2D Euclidean point cost; configurable global normalization (`none`, `path_length`, or `max_input_length`) and optional Sakoe-Chiba window. Default v1 oracle uses unnormalized cumulative cost and no window.
- `hausdorff`: maximum of the two directed point-set Hausdorff distances using projected Euclidean distance. Document that order and time are ignored.
- `discrete_frechet`: standard discrete coupling recurrence over projected Euclidean point distance.
- `lcss`: match when spatial distance is at most epsilon and optional index/time separation is within delta; expose epsilon/delta; canonical distance is `1 - lcss_length / min(n, m)`.
- `edr`: substitution cost zero within spatial epsilon and one otherwise, unit insert/delete; canonical distance is edit cost divided by `max(n, m)`.
- `erp`: projected-vector ERP with configurable gap point; default gap must be explicitly computed or configured per dataset and saved in the resolved config. Normalize only when the config requests it.

Edge cases must cover empty input rejection, one-point trajectories, repeated points, unequal lengths, identical trajectories, reversed trajectories, NaN rejection, timestamp absence, and extreme epsilon values.

### 8.3 Correctness hierarchy

For every classical measure:

1. hand-worked toy cases;
2. identity where mathematically required;
3. non-negativity;
4. symmetry when declared symmetric;
5. property-based finite-output tests;
6. parity against at least one independent trusted implementation or a clearly documented reference script over randomly generated small paths;
7. slow reference vs optimized implementation parity before enabling optimization.

Use tolerances justified by dtype and algorithm. Never weaken tests merely to match an optimized implementation.

## 9. Similarity notions and diagnostic semantics

Create `configs/notions/v1.yaml` as a versioned scientific artifact. Each notion has:

- `notion_id` and semantic version;
- human-readable definition and exclusions;
- relevant spatial, location, route, direction, temporal, sampling, and observation properties;
- expected outcome per transformation;
- tie tolerance or minimum margin policy;
- citations/decision notes;
- status: `active`, `experimental`, or `disabled`.

Initial notions:

- geometric shape;
- absolute geographic route;
- temporal dynamics;
- same underlying movement/observation equivalence;
- direction-aware movement;
- route/path structure.

Represent transformation expectations as `preserve`, `small_change`, `change`, `major_change`, `depends`, or `not_applicable`. Convert them to triplet labels only through an explicit task template. Keep contradictory notions as separate triplets, such as translated-copy vs nearby-different-shape under shape and absolute-route notions.

Triplet schema:

| Field | Meaning |
| --- | --- |
| `triplet_id` | Stable content-derived ID. |
| `query_id`, `candidate_a_id`, `candidate_b_id` | Canonical or variant IDs. |
| `notion_id`, `notion_version` | Declared interpretation. |
| `expected_order` | `a_closer`, `b_closer`, `tie`, `unspecified`. |
| `generator`, `generator_version` | Construction provenance. |
| `parameters_json`, `seed` | Exact regeneration inputs. |
| `quality_flags` | Constraint warnings. |
| `constraint_values_json` | Measured OD distances, overlap, detour ratio, etc. |

Triplet accuracy excludes `unspecified`, reports coverage, treats ties using the configured tolerance, and reports strict accuracy separately from tie-aware accuracy.

## 10. Perturbation engine

### 10.1 Contract

Every perturbation is a pure operation over a trajectory and seeded NumPy generator. It returns a new immutable trajectory plus provenance; it must not alter canonical arrays.

```python
class Perturbation(ABC):
    name: str
    version: str
    config_model: type[BaseModel]
    def apply(self, source: TrajectoryView, *, severity, rng) -> PerturbationResult: ...
```

Provenance includes variant ID, source ID, transformation, severity value and units, fully resolved parameters, seed, notion expectations, generator version, input hash, output hash, and quality flags. Variant IDs should be content-derived rather than assembled from unsafe free text.

All generated outputs must pass point, coordinate, and timestamp validation. If a requested transform is impossible for a short trajectory, return a typed `not_generated` result with a reason; do not create a degenerate example.

### 10.2 Required v1 transformations

- Independent GPS noise: 2D Gaussian noise in meters, sigma `{5,10,25,50,100}`.
- Correlated GPS drift: AR(1) error with configurable rho, default `0.9`, and a clearly documented stationary innovation scaling.
- Random point loss: retain `{1.0,0.9,0.75,0.5,0.25,0.1}`; optionally preserve endpoints; sample without replacement and restore original order.
- Contiguous outage: remove one interior segment of fraction `{0.05,0.10,0.20,0.30}`; define rounding and endpoint rules.
- Sampling-frequency reduction: time interval `{5,10,30,60,120}` seconds, spatial interval, or retention ratio; define deterministic first-point anchoring.
- Spatial quantization: grid widths `{10,25,50,100}` meters with grid origin recorded.
- Temporal jitter: seeded jitter with configured distribution/scale followed by a documented stable monotonic repair policy; report repair count.
- Speed distortion: global factors `{0.5,0.75,1.25,2.0}` and a later piecewise mode; preserve the first timestamp.
- Truncation: remove `{0.10,0.25,0.50}` from start, end, or symmetrically from both; define rounding.
- Reversal: reverse spatial order; for direction tests, choose and record whether timestamps are reversed, re-based, or omitted. Default movement reversal re-bases increasing timestamps while preserving segment durations in reverse order.
- Spatial translation: translate by magnitudes `{100,500,1000,2000,5000}` meters and a seeded/configured bearing; reject or flag translations outside dataset-valid bounds.
- Free-space detour: replace a valid interior segment using control points; record anchors, displacement, added length, and achieved detour ratio.
- Road-network detour: interface and disabled config only until map matching is an approved stretch phase.

### 10.3 Determinism tests

For each transformation verify:

- same input/config/seed gives byte-equivalent numeric output within serialization guarantees;
- different seeds alter stochastic outputs;
- source is unchanged;
- provenance regenerates the output;
- severity zero, where supported, is identity;
- timestamp order and coordinate validity are preserved;
- realized severity or constraint measurements are recorded.

## 11. Task and negative generators

### 11.1 Shared task contract

Each task consumes dataset IDs and a fully resolved config and emits immutable task artifacts before any method is evaluated. This decouples benchmark construction randomness from method randomness. Save task tables with content hashes and versions.

### 11.2 Oracle approximation

For each oracle independently:

1. create fixed query/database ID sets;
2. compute or load exact oracle distances;
3. exclude self matches;
4. produce deterministic rankings with ties broken by stable candidate ID after retaining tie metadata;
5. compare candidate-method rankings and, where meaningful, distance approximation.

Distance MAE/RMSE is opt-in and only valid after a declared calibration/normalization policy. Do not compare arbitrary learned similarity scales directly to oracle distances.

Cache oracle blocks by dataset hash, ID-set hash, oracle version/config, and dtype. Store completed block indexes so interrupted O(QN) work can resume.

### 11.3 Underlying-trajectory equivalence

Generate two or more observations from one source using odd/even samples, fixed-ratio samples, independent noise, contiguous gaps, and temporal jitter. Query with one view and mark other views of the same source as relevant. Ensure no source trajectory leaks between evaluation partitions. Include both random and hard negatives.

### 11.4 Counterfactual diagnostics

Implement generators for:

- downsampled source vs similar distinct candidate;
- low-noise vs high-noise source;
- small vs large detour;
- translated shape copy vs nearby different geometry, emitted under both shape and absolute-route notions;
- original/reversed comparisons for direction-aware and direction-invariant notions;
- spatially fixed time warp under spatial and temporal notions;
- same origin/destination but different route;
- high vs low partial route overlap with comparable OD differences.

Every generator must enforce and save numeric constraints. If it cannot find a qualifying candidate within a bounded search, save a rejection reason and continue; never relax thresholds silently.

### 11.5 Hard negatives

Define a common negative-generator result and implement:

- random;
- same OD / low route overlap;
- spatially nearby / geometrically different;
- translated shape copy;
- reversed trajectory;
- partial route overlap;
- same route / altered temporal behavior.

Put all thresholds in dataset/task configs: origin radius, destination radius, route-overlap definition, geographic-nearness band, shape-distance band, length-ratio band, and maximum candidates examined. Report yield and rejection rates. A hard-negative benchmark with low construction yield must fail its quality gate rather than quietly substituting random negatives.

For free-space v1, define route overlap using buffered projected polylines or resampled path coverage and version that definition. Do not call it road-segment overlap unless map matching exists.

### 11.6 Generalization and application tasks

Generalization modes are separate run dimensions: in-domain, user-held-out, temporal hold-out, zero-shot cross-dataset, limited adaptation, and full retraining. Validate feature and coordinate assumptions before permitting a cross-dataset run.

The application task consumes an external label/relevance provider with a version and provenance. Ship the interface and synthetic example in v1; keep real application tracks disabled until labels exist.

## 12. Retrieval and indexing

### 12.1 Exact classical retrieval

Implement correctness-first query-to-candidate scoring, NumPy `argpartition` for partial Top-K, then stable sorting of the selected set by `(distance, candidate_id)`. Support chunked candidates and process-level parallelism only after deterministic single-process correctness.

Persist all requested Top-K rows plus optional full-distance blocks. Avoid writing every QxN pair for large tiers unless the experiment explicitly requests it.

### 12.2 Embedding retrieval

Start with a NumPy exact L2/cosine implementation for adapter contract tests. Add FAISS FlatL2 after embeddings are validated. Record embedding normalization and distance choice. Only then add HNSW or another ANN index and compare it against FlatL2 on the same embeddings to isolate index approximation error.

Index artifacts must record dataset and ID order hashes, embedding hash/dimension/dtype, index type/config, library version, build hardware, build time, memory, and file size.

### 12.3 Relevance

Implement versioned relevance providers rather than embedding relevance logic in metrics. Providers may use same-source equivalence, oracle Top-K, graded oracle ranks, triplet expectations, or external labels. Store the provider name/version/config with every query result.

## 13. Evaluation semantics

### 13.1 Retrieval metrics

Implement per-query and aggregate Recall@K, Precision@K, Hit Rate@K, NDCG@K, and MRR. Validate `K`, handle queries with no relevant candidates using an explicit policy, and record that policy. NDCG must accept graded relevance; binary relevance is a special case. Macro-average over queries by default and retain per-query values for inference.

### 13.2 Ranking agreement

For common candidate sets, implement:

- Kendall tau-b to handle ties;
- Spearman rho with average ranks;
- Top-K Jaccard;
- Rank-Biased Overlap with configured persistence parameter;
- pairwise ordering agreement with a tie policy.

Record comparison depth/candidate universe. Do not compute a full-ranking correlation from two unrelated truncated Top-K lists. Build the method-by-method agreement matrix from per-query comparisons, then cluster on a documented distance such as `1 - mean_tau` only after addressing negative correlations and missing values.

### 13.3 Diagnostics and fingerprints

Compute per-notion strict and tie-aware triplet accuracy, valid-triplet count, coverage, bootstrap CI, and failure examples. A similarity fingerprint is a versioned collection of separately named diagnostic scores, not a hidden weighted average. Initially include sampling invariance, GPS-noise robustness, location sensitivity, shape sensitivity, direction sensitivity, temporal sensitivity, detour sensitivity, and same-OD hard-negative accuracy.

### 13.4 Robustness

Separate two concepts:

- task-performance robustness: quality metric `Q(alpha)` relative to clean `Q(0)`;
- pairwise sensitivity: change in distance from the clean trajectory.

For normalized robustness, handle `Q(0)=0` explicitly as undefined and report coverage. Compute robustness AUC with the trapezoidal rule after mapping tested severities to an ordered normalized [0,1] axis; store both raw curve points and summary. Never compare RAUC across transformations whose severity axes were not normalized/versioned consistently.

Monotonicity violation rate uses all ordered severity pairs for a source/method, a configurable numeric tolerance, and distance oriented so larger severity is expected not to reduce distance. Report per-source values, aggregate values, and the number of valid comparisons.

### 13.5 Hard-negative gap and generalization

Compute random and hard performance using matched queries, database size, K, and relevance policy. Define `delta_hard = random - hard`, with paired confidence intervals.

Compute generalization ratio only for metrics where division and direction are meaningful. For error/loss metrics use an explicitly named degradation transformation instead of reusing a quality ratio.

### 13.6 Statistical methodology

- Bootstrap 95% confidence intervals over query/source trajectories using a saved seed and enough resamples configured per tier.
- Use paired permutation as the default pairwise method test; optionally Wilcoxon where its assumptions and zero handling are documented.
- Apply Holm correction within a declared comparison family.
- Retain raw p-values, adjusted p-values, effect sizes, confidence intervals, sample sizes, and test configuration.
- For learned methods, aggregate three training seeds where feasible and report mean, standard deviation, and individual seeds. Do not pretend query-level bootstrap captures training-seed uncertainty.

## 14. Systems benchmark

Measure training, encoding, index construction, and online query as separate stages. Each systems run records:

- CPU model and logical/physical core counts;
- configured worker/thread counts and relevant thread environment variables;
- RAM, OS, Python, package versions;
- GPU model/driver/runtime when used;
- database size, query count, trajectory-length distribution;
- batch size, index type/config, embedding dimension/dtype;
- warm-up count, measured repetitions, synchronization policy;
- median and p95 query latency, throughput, and raw samples;
- peak process-tree CPU RSS and GPU memory;
- index and embedding sizes on disk;
- correctness/quality metric paired with the timing result.

Use `perf_counter_ns`; synchronize GPU operations before stopping timers. Run warm-ups separately. Avoid timing disk download or one-time dataset preparation as query latency. Store failures and timeouts as results, not missing rows.

Break-even analysis uses `Q* = C_E / (c_H - c_L)` only when `c_H > c_L`; otherwise report no finite break-even. State which one-time costs are included in `C_E` and provide uncertainty/sensitivity where timing variation matters.

## 15. Configuration model

The root experiment YAML must include:

- schema version;
- experiment ID, description, tags, and seed list;
- dataset version, split, scale tier, query/database selection;
- enabled methods and complete method configs;
- enabled tasks and task-specific settings;
- notion specification version;
- perturbations and ordered severity values with units;
- relevance providers and K values;
- metrics/statistical settings;
- systems settings;
- storage policy and output root;
- cache/resume policy;
- resource limits/timeouts.

Config loading pipeline:

1. load YAML;
2. resolve referenced dataset/method/notion/scale fragments;
3. expand defaults visibly;
4. validate cross-field constraints;
5. canonicalize to sorted JSON-compatible data;
6. hash the resolved config;
7. save `resolved_config.yaml` in the run before execution.

Reject unknown method names, unknown metrics, duplicate IDs, invalid severities, incompatible tasks, missing timestamps for temporal methods, and scale tiers larger than the available data.

Provide `trajbench config validate <file>` and `trajbench config resolve <file>` commands.

## 16. Orchestration, caching, and failure behavior

Model execution as explicit stages with durable status:

1. validate environment and config;
2. load and validate data/splits;
3. materialize or load task definitions;
4. fit/import method artifacts;
5. encode/build index where supported;
6. evaluate methods;
7. compute metrics/statistics;
8. commit Parquet outputs;
9. generate analysis artifacts if requested;
10. finalize manifest.

Each stage has input fingerprints, expected outputs, start/end times, and status. Write artifacts to a run-local temporary path, validate them, then rename atomically. On resume, reuse a stage only when input fingerprints and output checksums match. A `--force-stage` option may invalidate a specific stage and its dependents; never reuse stale downstream files.

The runner must:

- derive independent random streams from a root seed using `SeedSequence`;
- log structured events to console and JSONL;
- continue other method/task combinations when policy allows;
- preserve exception type, message, traceback, and stage in failure records;
- mark the run `complete`, `partial`, `failed`, or `cancelled`;
- finalize a manifest even after a handled failure;
- never label a partial run complete.

## 17. Artifact and result schemas

Use schema-versioned Parquet files with stable types. Minimum output layout:

```text
results/<experiment_id>/<run_id>/
|-- resolved_config.yaml
|-- manifest.json
|-- logs.jsonl
|-- tasks/
|   |-- queries.parquet
|   |-- triplets.parquet
|   `-- variants.parquet
|-- rankings.parquet
|-- query_metrics.parquet
|-- aggregate_metrics.parquet
|-- agreement.parquet
|-- robustness.parquet
|-- systems.parquet
|-- failures.parquet
|-- artifacts.json
`-- analysis/{figures,tables}/
```

Common columns for result tables: `schema_version`, `run_id`, `experiment_id`, `dataset`, `dataset_version`, `split`, `scale_tier`, `method`, `method_version`, `method_config_hash`, `task`, `task_version`, `seed`, and relevant IDs.

`rankings.parquet` additionally has query ID, candidate ID, rank (one-based), canonical distance, raw score, relevance value, ground-truth rank if applicable, and retrieval runtime.

`query_metrics.parquet` has query/source ID, metric name, K or null, value, valid flag, and reason if invalid.

`aggregate_metrics.parquet` has metric, value, CI bounds, sample size, aggregation policy, statistical seed, and grouping dimensions.

`robustness.parquet` has source/query ID, perturbation, severity value/unit/index, clean value, perturbed value, normalized value, and valid flag.

`systems.parquet` stores stage, raw timing sample or summary type, timing/memory/storage values, workload dimensions, and hardware fingerprint.

The manifest includes code commit and dirty flag when Git exists, package version, resolved config hash, dataset checksums, task hashes, method versions, hardware/software inventory, start/end timestamps in UTC, artifact paths/checksums/row counts, status, warnings, and failures.

Create DuckDB views over a result root on demand. Analysis code must tolerate multiple schema versions only through explicit migrations or compatibility views.

## 18. External learned-baseline protocol

The core orchestrator communicates with each isolated baseline through files and a subprocess/container boundary.

Input contract:

```text
request.json
dataset_path
train_ids.npy
val_ids.npy
query_ids.npy
database_ids.npy
config.json
```

`request.json` declares protocol version, operation (`fit_encode`, `encode`, `rank`, or `distance`), mounted paths, expected outputs, seed, and resource constraints.

Output contract:

```text
status.json
metadata.json
timings.json
embeddings.npy       # optional
rankings.parquet     # optional
distances.parquet    # optional
stdout.log
stderr.log
```

Validate all outputs: IDs, row counts, embedding dimension/dtype/finite values, rank uniqueness/order, score direction, file checksums, and protocol version. Capture the upstream repository commit, local patches, model checkpoint hash, environment lock, license, training data, hyperparameters, and citation.

Integration order:

1. build a fake external adapter fixture;
2. prove protocol round-trip and failure handling;
3. select one mature baseline based on reproducibility and dataset compatibility;
4. integrate it without modifying core benchmark logic;
5. select a second method from a different family;
6. only then consider SIMformer, MovSemCL, LB-TrajRep, or other modern methods after verifying source and license.

Do not promise a named learned method until its repository, weights/data assumptions, and license have been audited.

## 19. CLI contract

Required commands:

```text
trajbench prepare --dataset porto --config configs/datasets/porto.yaml
trajbench validate-data data/processed/porto/v1
trajbench perturb --dataset ... --trajectory ... --type ... --severity ... --seed ... --output ...
trajbench run configs/experiments/core_porto.yaml [--resume] [--dry-run]
trajbench analyze --experiment ... --results-root results --output ...
trajbench dashboard --results-root results
trajbench config validate <file>
trajbench config resolve <file>
trajbench methods list
trajbench datasets list
trajbench results validate <run-dir>
```

Every mutating command supports `--dry-run` where meaningful and prints resolved input/output paths. Errors must say how to fix the problem. Commands return nonzero on invalid data/config, failed required stages, or invalid artifacts. `run --dry-run` prints the stage DAG, estimated pair counts, cache hits, and obvious resource risks without executing measures.

## 20. Analysis outputs

Generate analysis solely from Parquet plus manifests. Each figure/table generator accepts an input result root and output directory and writes a small provenance sidecar.

Required figures:

1. benchmark architecture (static documentation asset);
2. counterfactual trajectory examples;
3. method agreement heatmap and clustering;
4. similarity fingerprint heatmap;
5. robustness curves with uncertainty;
6. random vs hard-negative paired comparison;
7. cross-domain comparison when data exists;
8. accuracy-latency Pareto frontier;
9. break-even total-cost curves;
10. selected dashboard workflow screenshot for publication.

Required tables:

- dataset statistics and licensing status;
- method taxonomy/configurations;
- benchmark tracks and relevance policies;
- per-oracle retrieval results;
- diagnostic triplet results and coverage;
- robustness and MVR summaries;
- hard-negative results and construction yields;
- cross-domain results if enabled;
- systems costs and workload details;
- prespecified ablations.

Keep placeholder or unavailable cells explicit. Never convert missing experiments to zero.

## 21. Dashboard implementation

The dashboard reads manifests and Parquet through cached service functions. Shared services may call core library functions for small interactive examples, but pages contain no benchmark formulas or data-construction logic.

1. Dataset Explorer: coverage map, path-length/duration/sampling distributions, modes, trajectory detail, dataset/version/license badges.
2. Pair Explorer: select two IDs; overlay projected/WGS84 paths, start/end/sample points, timestamps, length, duration, speed profile, per-method distance and runtime.
3. Counterfactual Laboratory: apply a small deterministic transformation, show source vs variant, resolved controls/provenance, and distance changes. Limit input size and methods to avoid freezing the UI.
4. Retrieval Disagreement Explorer: side-by-side Top-K lists, shared/unique candidates, disagreement scores, hard-negative labels, clickable overlays.
5. Similarity Fingerprints: method-by-dimension heatmap with dataset/method/length/mode/severity filters, confidence intervals, sample sizes, and coverage.
6. Robustness Laboratory: quality or sensitivity vs severity, transformation/method filters, RAUC/MVR summaries, confidence bands.
7. Accuracy-Efficiency Explorer: Pareto plot across quality, p95 latency, throughput, memory, index size, robustness, database size, and training/encoding cost.
8. Experiment Builder: form backed by the same Pydantic config model; validate and download YAML. Executing the generated experiment from the UI is optional and disabled by default for safety.

Add empty/error/loading states, result schema validation, source run links, and export of filtered CSV/PNG where practical. Test service functions normally and use a small smoke test for page imports; do not rely only on manual clicks.

## 22. Phased implementation backlog

### Phase 0 - Bootstrap and contracts

Deliver:

- package skeleton, `pyproject.toml`, CLI shell, lint/type/test config;
- ADR template and initial decisions from Section 3;
- Pydantic config skeleton with schema versioning;
- synthetic fixture generator and core data dataclasses;
- CI workflow running lint, types, unit tests on CPU.

Exit criteria:

- editable install works;
- `trajbench --help` and `trajbench config validate` work;
- all quality commands pass;
- no real dataset or optional GPU dependency is required.

### Phase 1 - Canonical data layer

Deliver:

- processed data writer/reader, memory-mapped views, schema validation;
- CRS projection/inverse projection;
- metadata/statistics/checksum generation;
- deterministic splitting and scale selection;
- synthetic preparation end to end;
- Porto and GeoLife loaders plus acquisition/license docs;
- disabled Germany gate.

Exit criteria:

- repeated preparation with identical input/config yields identical content hashes;
- corruption tests prove validator failures;
- Porto/GeoLife sample fixtures pass without committing raw data;
- user-held-out split has zero user overlap.

### Phase 2 - Classical measures

Deliver:

- base API, capabilities, registry, config models;
- seven required classical methods;
- toy, property, and reference tests;
- a small pairwise timing harness that is not yet the formal systems benchmark.

Exit criteria:

- the proposal's `for measure in registry: measure.distance(q, c)` use case works through the finalized API;
- correctness suite passes for every method;
- method configs and definitions are documented;
- optimized code, if any, matches its reference implementation.

### Phase 3 - Perturbations, notions, and provenance

Deliver:

- notion v1 YAML and validation;
- pure perturbation API and all free-space v1 transforms;
- variant/provenance schemas and regeneration command;
- CLI `perturb`;
- determinism and invariant tests.

Exit criteria:

- the same trajectory/type/severity/seed regenerates the same variant;
- all variants validate;
- provenance captures realized parameters and hashes;
- unsupported transforms fail clearly.

### Phase 4 - Task construction and negatives

Deliver:

- immutable task artifacts;
- oracle, equivalence, diagnostic, and retrieval tasks;
- random and all v1 hard-negative generators;
- relevance providers and constraint/yield reports;
- task quality validation.

Exit criteria:

- all diagnostic families can generate valid synthetic examples;
- opposite expectations can coexist under different notions;
- hard-negative thresholds and achieved values are saved;
- task construction is deterministic and independent of method order.

### Phase 5 - Retrieval, metrics, and statistics

Deliver:

- exact chunked classical Top-K;
- retrieval, agreement, diagnostic, robustness, MVR, hard-gap metrics;
- bootstrap CI, paired permutation, Holm correction;
- fingerprint construction;
- comprehensive metric edge-case tests.

Exit criteria:

- hand-computed examples match;
- full vs chunked retrieval rankings match;
- tie and empty-relevance policies are tested and recorded;
- analysis never mixes incomparable candidate universes or notions.

### Phase 6 - Runner and storage

Deliver:

- full configuration resolution and hashing;
- staged runner, cache/resume, failure records, structured logs;
- Parquet writers, manifest, artifact validation, DuckDB views;
- Tiny synthetic experiment config;
- `run`, `analyze`, and `results validate` commands.

Exit criteria:

- `trajbench run configs/ci/tiny_synthetic.yaml` succeeds on CPU;
- required Parquet files and manifest validate;
- interrupted run resumes without recomputing valid stages;
- changed input/config invalidates the right cache descendants;
- handled method failure yields `partial`, not `complete`.

### Phase 7 - Core experiments and analysis

Deliver:

- Porto/GeoLife core configs at Tiny and Standard tiers;
- precomputation/block caching for expensive oracles;
- all required core tables and figures;
- experiment provenance sidecars;
- pilot validation of task yields and thresholds.

Exit criteria:

- results answer RQ1-RQ3 and hard-negative questions without unsupported claims;
- every number in a generated table traces to a run and config;
- paper figures rebuild in a clean output directory;
- pilot findings lead to versioned config changes, never silent edits.

### Phase 8 - Dashboard

Deliver pages in this order: Dataset, Pair, Counterfactual, Disagreement, Robustness, Fingerprints, Efficiency, Builder.

Exit criteria:

- large experiment views perform no benchmark recomputation;
- pages surface run/version/sample-size/CI information;
- invalid or partial runs are visibly labeled;
- dashboard starts against Tiny saved results and page smoke tests pass.

### Phase 9 - Learned adapters

Deliver:

- external protocol and fake adapter;
- reproducibility/license audit template;
- first approved baseline container/adapter;
- second approved baseline from a distinct method family;
- in-domain evaluation and three seeds where feasible.

Exit criteria:

- both baselines run without changing central task/evaluation logic;
- output validator catches malformed rankings/embeddings;
- upstream commit, patch, lock, checkpoint, and config are traceable;
- core CPU benchmark remains installable without learned extras.

### Phase 10 - Generalization, indexing, and formal systems study

Deliver:

- NumPy exact embedding retrieval and FAISS FlatL2 parity;
- optional HNSW with exact-index ablation;
- user-held-out/temporal/cross-dataset modes where compatible;
- standardized profiling and break-even/Pareto analysis;
- systems runs over 1K, 10K, 50K, and 100K+ as resources permit.

Exit criteria:

- representation error is separated from ANN error;
- workload and hardware metadata are complete;
- zero-shot/adaptation/retraining are never pooled;
- break-even results state included costs and no-finite-break-even cases.

### Phase 11 - Release hardening

Deliver:

- installation, dataset acquisition, extension, and reproducibility docs;
- security/privacy/license review;
- clean-checkout rehearsal;
- final example configs and expected Tiny checksums;
- versioned release, citation metadata, changelog, and archival plan.

Exit criteria:

- a fresh environment can install, validate data, run Tiny, analyze, and launch the dashboard from documented commands;
- no raw restricted data, secrets, local absolute paths, or oversized generated results are committed;
- claims in prose are linked to generated evidence and limitations;
- all required licenses and citations are present.

## 23. Test matrix

| Layer | Required tests |
| --- | --- |
| Config | Unknown keys, composition, cross-field incompatibilities, stable resolved hash, migration rejection. |
| Data | Round trip, mmap slicing, offsets, IDs, CRS, timestamps, metadata consistency, checksums, splits. |
| Measures | Hand cases, properties, reference parity, edge cases, optimized/reference equivalence. |
| Perturbations | Seed determinism, source immutability, zero severity, validity, provenance regeneration, realized severity. |
| Tasks | Determinism, no leakage/self-match, constraint satisfaction, bounded failure, yield reporting. |
| Retrieval | Stable ties, Top-K correctness, chunk parity, exclusion rules, cache keys. |
| Metrics | Hand calculations, ties, no relevance, graded relevance, missing groups, CI determinism, correction families. |
| Storage | Schema, round trip, atomic writes, manifest/artifact checksums, partial/failure runs, compatibility rejection. |
| Adapters | Good fixture, timeout, nonzero exit, missing output, bad IDs/ranks, NaN embeddings, protocol mismatch. |
| CLI | Help, success/failure codes, dry run, actionable errors, resolved paths. |
| Dashboard | Service queries, partial/empty state, page import smoke, config generation round trip. |
| End to end | Prepare synthetic -> build tasks -> run methods -> store -> analyze -> dashboard-read. |

Run slow/reference/integration tests behind markers, but run the Tiny end-to-end smoke in CI. Add regression fixtures only when their origin and expected values are documented.

## 24. Reproducibility, privacy, and licensing

- Keep raw/interim/processed data and ordinary results out of Git by default.
- Version download/preparation scripts, split IDs where licensing permits, configs, and checksums.
- Use stable UTC timestamps in manifests but exclude volatile timestamps from content fingerprints.
- Record Git commit and dirty status; when no Git metadata exists, record package/source fingerprint and mark it explicitly.
- Never log full sensitive trajectories by default.
- Treat user IDs as pseudonymous and avoid identity inference.
- Show dataset license and redistribution status in docs and dashboard.
- Prefer scripts and IDs/checksums over data redistribution.
- Require explicit approval before enabling user-uploaded datasets or human annotations.
- Sanitize config-derived paths and IDs; keep run outputs under the configured result root.
- Do not load arbitrary pickle files from datasets or external baselines.

## 25. Performance and resource safeguards

Before execution, estimate pair count, expected dynamic-programming cell count, and approximate result size. Warn or require an explicit large-run flag above configurable limits. At Standard tier, use chunking, memory mapping, and resumable oracle blocks. At larger tiers, do not run exact O(QN) dynamic-programming retrieval for every measure unless the experiment explicitly budgets it.

Cache only deterministic artifacts with complete fingerprints. Bound worker counts and prevent NumPy/BLAS oversubscription. Record timeouts instead of hanging indefinitely. Keep accuracy experiments and scalability experiments separate so pruning, sampling, or ANN does not silently change the scientific question.

## 26. Risks and gates

| Risk | Gate or mitigation |
| --- | --- |
| Ambiguous similarity definitions | Version notions and expectations; report unspecified coverage; require ADR for changes. |
| Classical implementation errors | Reference implementation parity before optimization or publication runs. |
| Exact retrieval cost | Scale tiers, dry-run estimates, chunking, block cache, separate systems experiments. |
| Hard-negative scarcity | Constraint/yield reports and minimum yield gates; no silent random fallback. |
| Dataset leakage | Source/user/time-aware split validators and same-source grouping. |
| Dataset license uncertainty | Disabled config until source/license approval; distribute scripts rather than restricted data. |
| External baseline dependency rot | Isolated locked containers, fake-protocol tests, upstream commit/patch recording. |
| Dashboard scope creep | Implement after saved-result pipeline; no large compute in UI. |
| Misleading aggregate score | Results per oracle/notion/transformation; fingerprints remain vectors. |
| Invalid statistical confidence | Pair by query/source, expose sample sizes, separate training-seed and query uncertainty. |
| Timing confounds | Warm-up/repetition/synchronization/thread/hardware protocol and raw timing retention. |

## 27. Questions requiring human/supervisor decisions

These do not block Phases 0-6 unless noted:

1. Which exact “Germany” dataset is intended, and what is its source/license? This blocks that loader and third-dataset claims.
2. Which two learned baselines are mandatory after the reproducibility audit? This blocks named Phase 9 integrations, not the adapter protocol.
3. What exact geometric definition and thresholds should be canonical for route overlap and same-OD hard negatives on each dataset? Synthetic defaults can be implemented, but publication configs require review.
4. Should the canonical DTW oracle be raw cumulative cost or normalized cost? This plan chooses raw cumulative cost for v1; changing it requires a new oracle version.
5. Which application-level labels, if any, will support the Application track? Without them, keep the track as an interface and synthetic example.
6. What compute budget/hardware is available for Standard and larger tiers? This determines feasible methods, seeds, and QxN workloads.
7. Are maps allowed to display raw public-dataset paths, or should the dashboard apply optional spatial obfuscation for demos?
8. Will human preference judgments be collected? If yes, ethics review and a separate study protocol are prerequisites.

## 28. Final acceptance checklist

- [ ] Base install and optional extras are documented and locked.
- [ ] Tiny CPU CI passes lint, types, tests, full run, artifact validation, and analysis.
- [ ] At least Porto and GeoLife prepare reproducibly from documented raw inputs.
- [ ] Canonical data, split, task, config, result, and adapter schemas are versioned.
- [ ] Seven classical methods pass hand, property, and reference tests.
- [ ] All free-space v1 perturbations are deterministic and provenance-complete.
- [ ] Diagnostic triplets are notion-specific and report coverage/constraint quality.
- [ ] Random and hard negatives use matched evaluation and report construction yield.
- [ ] Retrieval, agreement, robustness, monotonicity, and statistical metrics pass hand tests.
- [ ] Runner resumes safely and invalidates stale caches correctly.
- [ ] Manifests trace code, config, data, methods, hardware, artifacts, warnings, and failures.
- [ ] Dashboard consumes saved artifacts and clearly labels partial/invalid runs.
- [ ] At least two audited learned baselines use the external protocol for the publication target.
- [ ] Exact embedding retrieval precedes ANN comparisons.
- [ ] Cross-domain modes and systems results state their assumptions and workload details.
- [ ] Paper figures/tables regenerate from saved results with provenance.
- [ ] No universal aggregate leaderboard is produced.
- [ ] No restricted data, secrets, local paths, or unsupported scientific claims are released.

## 29. Suggested execution prompt for a lower-cost coding model

Give the implementing model this repository and this instruction:

> Implement TrajSimBench by following `IMPLEMENTATION_PLAN.md` phase by phase. Start with the earliest incomplete phase only. Inspect the repository before editing, preserve existing user changes, and read any `AGENTS.md` files. Do not skip exit criteria, do not implement later-phase features early, and do not invent scientific definitions or dataset sources. Make the smallest coherent batch of changes that completes or materially advances the current phase, run the prescribed checks, fix failures, and report changed files, commands run, results, remaining checklist items, and any decision that requires the user or supervisor. Use `apply_patch` for edits. Keep the CPU-only Tiny benchmark working at all times.

For best reliability, invoke the model once per phase or per small group of closely related deliverables rather than asking it to implement the entire benchmark in one turn.
