"""Cached read-only result access used by every dashboard page."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from trajsimbench.storage.artifacts import validate_run_directory
from trajsimbench.storage.manifest import load_manifest
from trajsimbench.storage.parquet import read_parquet


@lru_cache(maxsize=32)
def list_runs(results_root: str | Path) -> list[dict[str, Any]]:
    root = Path(results_root)
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = load_manifest(manifest_path.parent)
        except Exception as exc:
            runs.append(
                {
                    "run_dir": str(manifest_path.parent),
                    "run_id": None,
                    "status": "invalid",
                    "error": str(exc),
                }
            )
        else:
            runs.append(
                {
                    "run_dir": str(manifest_path.parent),
                    "run_id": manifest.get("run_id"),
                    "experiment_id": manifest.get("experiment_id"),
                    "status": manifest.get("status"),
                    "resolved_config_hash": manifest.get("resolved_config_hash"),
                }
            )
    return runs


@lru_cache(maxsize=128)
def read_result_table(run_dir: str | Path, table: str) -> list[dict[str, Any]]:
    root = Path(run_dir)
    path = (
        root
        / ("tasks" if table in {"queries", "triplets", "variants"} else "")
        / f"{table}.parquet"
    )
    if not path.exists():
        return []
    return read_parquet(path)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    return validate_run_directory(Path(run_dir), require_all=False)


def filter_rows(rows: Sequence[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(value is None or row.get(key) == value for key, value in filters.items())
    ]
