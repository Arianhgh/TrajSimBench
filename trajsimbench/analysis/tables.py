"""CSV report generation from authoritative Parquet result artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .pareto import pareto_frontier

TABLE_NAMES = (
    "dataset_statistics",
    "retrieval_metrics",
    "agreement_metrics",
    "robustness_metrics",
    "systems",
    "method_summary",
    "run_summary",
    "ranking_samples",
    "pareto_summary",
    "reproducibility_manifest",
    "parameters",
)


def _parquet_paths(root: Path, table: str) -> list[Path]:
    return sorted(root.rglob(f"{table}.parquet"))


def load_result_rows(root: str | Path, table: str) -> list[dict[str, Any]]:
    """Read every matching Parquet table below ``root`` in stable path order."""
    rows: list[dict[str, Any]] = []
    for path in _parquet_paths(Path(root), table):
        frame = pd.read_parquet(path)
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)
    return path


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"path": str(path), **payload})
    return rows


def _method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or "method" not in rows[0]:
        return []
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return frame[["method"]].drop_duplicates().to_dict(orient="records")
    return frame.groupby("method", dropna=False)[numeric].mean(numeric_only=True).reset_index().to_dict(
        orient="records"
    )


def generate_tables(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Create a fixed, machine-readable set of report tables.

    Empty source tables still produce an empty CSV. This makes missing benchmark
    outputs visible without making analysis depend on a particular experiment.
    """
    root = Path(results_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source = {
        "retrieval_metrics": load_result_rows(root, "aggregate_metrics"),
        "agreement_metrics": load_result_rows(root, "agreement"),
        "robustness_metrics": load_result_rows(root, "robustness"),
        "systems": load_result_rows(root, "systems"),
        "ranking_samples": load_result_rows(root, "rankings"),
    }
    manifests = _manifest_rows(root)
    dataset_rows = [
        {key: value for key, value in row.items() if key in {"dataset", "dataset_version"}}
        for row in manifests
    ]
    dataset_rows = [row for row in dataset_rows if row]
    method_rows = _method_summary(source["retrieval_metrics"])
    pareto_rows = pareto_frontier(source["systems"])
    parameters_rows = [{"key": str(key), "value": json.dumps(value, sort_keys=True)} for key, value in (parameters or {}).items()]

    content: dict[str, list[dict[str, Any]]] = {
        "dataset_statistics": dataset_rows,
        **source,
        "method_summary": method_rows,
        "run_summary": manifests,
        "pareto_summary": pareto_rows,
        "reproducibility_manifest": manifests,
        "parameters": parameters_rows,
    }
    return {name: _write_csv(output / f"{name}.csv", content[name]) for name in TABLE_NAMES}
