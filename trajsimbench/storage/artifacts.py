"""Artifact checks and safe run-directory helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import load_manifest, sha256_file
from .parquet import read_parquet
from .schemas import SCHEMA_VERSION, TABLE_SCHEMAS, validate_rows

REQUIRED_RUN_FILES = (
    "resolved_config.yaml",
    "manifest.json",
    "rankings.parquet",
    "query_metrics.parquet",
    "aggregate_metrics.parquet",
    "agreement.parquet",
    "robustness.parquet",
    "systems.parquet",
    "failures.parquet",
    "fingerprints.parquet",
    "artifacts.json",
)


def artifact_fingerprint(paths: list[Path] | tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def write_artifacts(run_dir: Path, entries: list[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "artifacts.json"
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "artifacts": [dict(entry) for entry in entries]},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def validate_run_directory(run_dir: Path, *, require_all: bool = True) -> dict[str, Any]:
    root = Path(run_dir)
    errors: list[str] = []
    if not root.exists():
        return {
            "valid": False,
            "errors": [f"run directory does not exist: {root}"],
            "status": "missing",
        }
    manifest = None
    try:
        manifest = load_manifest(root)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append("manifest schema_version is unsupported")
    except Exception as exc:
        errors.append(f"manifest: {exc}")
    if require_all:
        errors.extend(
            f"missing required artifact: {name}"
            for name in REQUIRED_RUN_FILES
            if not (root / name).exists()
        )
    for table_name in TABLE_SCHEMAS:
        path = (
            root
            / ("tasks" if table_name in {"queries", "triplets", "variants"} else "")
            / f"{table_name}.parquet"
        )
        if path.exists():
            try:
                report = validate_rows(read_parquet(path), table=table_name)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
            else:
                errors.extend(f"{table_name}: {error}" for error in report["errors"])
    return {
        "valid": not errors,
        "errors": errors,
        "status": (manifest or {}).get("status", "unknown"),
        "manifest": manifest,
    }
