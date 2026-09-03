# TrajSimBench

TrajSimBench is a reproducible benchmark for comparing trajectory similarity
notions. The foundation is CPU-only: canonical trajectory data, strict YAML
configuration, deterministic preparation, projections, splits, and validation
are usable without a GPU or restricted raw data.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

For a pinned CPU environment, install `requirements.lock` after creating the
environment:

```powershell
python -m pip install -r requirements.lock
```

The dashboard and FAISS extras are optional; PyTorch and any learned baseline
remain outside the core environment.

The complete Tiny workflow is:

```powershell
python -m trajsimbench.cli config validate configs/ci/tiny_synthetic.yaml
python -m trajsimbench.cli run configs/ci/tiny_synthetic.yaml
python -m trajsimbench.cli results validate results/tiny_synthetic_cpu/<run-id>
python -m trajsimbench.cli analyze --experiment tiny_synthetic_cpu --results-root results --output analysis
python -m trajsimbench.cli dashboard --results-root results
```

The equivalent script entry points live under `scripts/`.  `scripts/download`
prints source/license instructions only; it never guesses or redistributes a
restricted raw mobility dataset.

The repository does not redistribute Porto, GeoLife, or any other restricted
mobility data. Follow [the dataset guide](docs/adding-a-dataset.md) and place
user-supplied raw files below `data/raw/`.

The synthetic fixture can be prepared entirely locally through
`trajsimbench.data.loaders.synthetic.prepare_synthetic`. Later benchmark and
CLI layers consume the same canonical directory format documented in
[`docs/data-format.md`](docs/data-format.md).

## Design commitments

- WGS84 longitude/latitude are preserved; metric operations use the resolved
  projected CRS and meters.
- Canonical point arrays are `float64`, memory-mappable, and exposed read-only.
- Distances use lower-is-more-similar semantics.
- Configuration models reject unknown keys and retain a stable resolved hash.
- Raw mobility data, generated results, and local paths are ignored by Git.

See `IMPLEMENTATION_PLAN.md` for the full benchmark roadmap.
