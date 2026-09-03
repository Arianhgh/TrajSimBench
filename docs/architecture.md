# Architecture

TrajSimBench is layered around immutable, versioned artifacts. Raw inputs are
read by isolated dataset loaders; loaders produce a canonical processed
directory; later benchmark stages consume `TrajectoryDataset` views without
knowing the source format.

```text
raw input -> loader -> projected canonical dataset -> tasks/measures -> results
                 \-> inspection/rejection report       \-> checksums/manifest
```

The foundation deliberately keeps data preparation separate from measures,
retrieval, evaluation, storage, orchestration, and the CLI. The public data
API is usable from Python in a CPU-only environment. `points.npy` is always
`float64`, while `TrajectoryDataset.open(..., mmap=True)` uses a read-only
memory map by default.

Every prepared version is written into a temporary sibling directory, fully
validated, checksummed, and then moved into place. An existing valid version is
idempotently reusable; a different version is never silently overwritten.

Configuration follows the same contract: YAML fragments are resolved first,
then strict Pydantic models reject unknown keys and cross-field errors, and a
sorted JSON representation supplies the stable configuration hash.
