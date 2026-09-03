"""Run manifest creation, checksums, and portable provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trajsimbench.utils.hardware import hardware_info

from .schemas import SCHEMA_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or ".")
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
    }


@dataclass
class ManifestBuilder:
    run_id: str
    experiment_id: str
    output_root: Path
    dataset_checksums: dict[str, str] = field(default_factory=dict)
    task_hashes: dict[str, str] = field(default_factory=dict)
    method_versions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def artifact_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.output_root.rglob("*")):
            if not path.is_file() or path.name in {"manifest.json", "artifacts.json"}:
                continue
            entries.append(
                {
                    "path": str(path.relative_to(self.output_root)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": _row_count(path),
                }
            )
        return entries

    def build(
        self, *, resolved_config_hash: str | None = None, root: Path | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "code": code_provenance(root or self.output_root),
            "resolved_config_hash": resolved_config_hash,
            "dataset_checksums": self.dataset_checksums,
            "task_hashes": self.task_hashes,
            "method_versions": self.method_versions,
            "hardware": {**hardware_info(), "python_runtime": sys.version},
            "artifacts": self.artifact_entries(),
            "warnings": self.warnings,
            "failures": self.failures,
        }

    def write(self, *, resolved_config_hash: str | None = None, root: Path | None = None) -> Path:
        path = self.output_root / "manifest.json"
        path.write_text(
            json.dumps(
                self.build(resolved_config_hash=resolved_config_hash, root=root),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return path


def _row_count(path: Path) -> int | None:
    if path.suffix != ".parquet":
        return None
    try:
        from .parquet import read_parquet

        return len(read_parquet(path))
    except Exception:
        return None


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    run_id: str,
    experiment_id: str,
    output_root: Path,
    *,
    status: str = "complete",
    resolved_config_hash: str | None = None,
) -> dict[str, Any]:
    builder = ManifestBuilder(run_id, experiment_id, Path(output_root), status=status)
    return builder.build(resolved_config_hash=resolved_config_hash)
