# External baseline adapter protocol

TrajSimBench keeps learned or representation-based baselines outside the core
environment. An adapter is a file-level subprocess boundary and may run in a
container, a virtual environment, or a separately managed host.

## Request

The runner creates a directory containing:

```text
request.json       # protocol_version, operation, seed, expected_outputs
dataset_path       # read-only processed dataset mount
train_ids.npy
val_ids.npy
query_ids.npy
database_ids.npy
config.json
```

`operation` is one of `fit_encode`, `encode`, `rank`, or `distance`. Paths in
the request are absolute inside the adapter environment and must not be
reinterpreted as host paths. The adapter must not write outside its output
directory.

## Response

The adapter writes:

```text
status.json        # {protocol_version, status, error?}
metadata.json      # method/version/checkpoint/license/config provenance
timings.json       # stage timings in nanoseconds and resource samples
embeddings.npy     # optional, finite [row_count, dimension]
rankings.parquet   # optional, unique one-based rank per query
distances.parquet  # optional, lower-is-better canonical distance
stdout.log
stderr.log
```

`status` is `complete` or `failed`; a failure is a result with its exception,
traceback, and logs retained. A successful response is accepted only when the
protocol version, IDs, row counts, finite values, embedding dimension/dtype,
rank uniqueness/order, and score direction validate. The core never imports
the adapter's Python modules.

## Provenance requirements

`metadata.json` records the upstream repository commit, local patches,
checkpoint hash, environment lock, license, training data, hyperparameters,
and citation. The core benchmark stores these values with method artifacts so
results cannot be mistaken for an untracked implementation.

## CPU MVP boundary

The MVP defines and validates this protocol but does not train or download a
learned baseline. The `baseline_envs/fake` runner is a deterministic fixture
for round-trip and malformed-output tests. Installing PyTorch, a GPU runtime,
or an external research repository is not required for the Tiny benchmark.

