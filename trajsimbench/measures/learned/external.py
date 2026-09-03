"""Validated subprocess protocol for optional learned baselines.

This module is deliberately model-agnostic.  It creates a request directory,
executes an adapter command without a shell, retains stdout/stderr, and
validates the files described in ``docs/baseline-adapter-protocol.md``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from trajsimbench.storage.parquet import read_parquet

ADAPTER_PROTOCOL_VERSION = "1.0"
ADAPTER_OPERATIONS = frozenset({"fit_encode", "encode", "rank", "distance"})


class ExternalAdapterError(RuntimeError):
    """Raised when an external adapter violates the file protocol."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class AdapterRequest:
    """Input manifest supplied to an isolated adapter process."""

    operation: str
    output_dir: Path
    dataset_path: Path | None = None
    train_ids: Path | None = None
    val_ids: Path | None = None
    query_ids: Path | None = None
    database_ids: Path | None = None
    config_path: Path | None = None
    expected_outputs: tuple[str, ...] = ()
    seed: int = 0
    resource_limits: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = ADAPTER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != ADAPTER_PROTOCOL_VERSION:
            raise ExternalAdapterError(
                f"unsupported adapter protocol {self.protocol_version!r}; "
                f"expected {ADAPTER_PROTOCOL_VERSION!r}"
            )
        if self.operation not in ADAPTER_OPERATIONS:
            raise ExternalAdapterError(
                f"operation must be one of {sorted(ADAPTER_OPERATIONS)}, got {self.operation!r}"
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        for name in (
            "dataset_path",
            "train_ids",
            "val_ids",
            "query_ids",
            "database_ids",
            "config_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        object.__setattr__(self, "expected_outputs", tuple(map(str, self.expected_outputs)))
        object.__setattr__(self, "resource_limits", dict(self.resource_limits))

    @property
    def path(self) -> Path:
        return self.output_dir / "request.json"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "operation": self.operation,
            "output_dir": self.output_dir,
            "expected_outputs": self.expected_outputs,
            "seed": self.seed,
            "resource_limits": self.resource_limits,
        }
        paths = {
            "dataset_path": self.dataset_path,
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "query_ids": self.query_ids,
            "database_ids": self.database_ids,
            "config_path": self.config_path,
        }
        value.update({key: path for key, path in paths.items() if path is not None})
        return _jsonable(value)

    def write(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.path

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdapterRequest:
        allowed = {
            "protocol_version",
            "operation",
            "output_dir",
            "dataset_path",
            "train_ids",
            "val_ids",
            "query_ids",
            "database_ids",
            "config_path",
            "expected_outputs",
            "seed",
            "resource_limits",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ExternalAdapterError(f"unknown request fields: {unknown}")
        data = dict(value)
        if "output_dir" not in data:
            raise ExternalAdapterError("request.output_dir is required")
        return cls(**data)

    @classmethod
    def read(cls, path: Path) -> AdapterRequest:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalAdapterError(f"cannot read adapter request {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ExternalAdapterError("adapter request must be a JSON object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class AdapterRunResult:
    """Retained subprocess outcome and output validation report."""

    status: str
    returncode: int | None
    elapsed_ns: int
    output_dir: Path
    validation: dict[str, Any]
    timed_out: bool = False

    @property
    def valid(self) -> bool:
        return bool(self.validation.get("valid", False)) and self.status == "complete"


def _load_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ExternalAdapterError(f"cannot read ID array {path}: {exc}") from exc
    if values.ndim != 1:
        raise ExternalAdapterError(f"ID array must be one-dimensional: {path}")
    return [str(value) for value in values.tolist()]


def _validate_rank_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_by_query: dict[str, tuple[set[str], set[int]]] = {}
    for index, row in enumerate(rows):
        for column in ("query_id", "candidate_id", "rank"):
            if column not in row:
                errors.append(f"rankings row {index} is missing {column}")
        if "query_id" not in row or "candidate_id" not in row or "rank" not in row:
            continue
        query_id = str(row["query_id"])
        candidate_id = str(row["candidate_id"])
        try:
            rank = int(row["rank"])
        except (TypeError, ValueError):
            errors.append(f"rankings row {index} rank is not an integer")
            continue
        ids, ranks = seen_by_query.setdefault(query_id, (set(), set()))
        if candidate_id in ids:
            errors.append(f"duplicate candidate {candidate_id!r} for query {query_id!r}")
        if rank < 1 or rank in ranks:
            errors.append(f"invalid or duplicate rank {rank!r} for query {query_id!r}")
        ids.add(candidate_id)
        ranks.add(rank)
        if "distance" in row:
            try:
                distance = float(row["distance"])
            except (TypeError, ValueError):
                errors.append(f"rankings row {index} distance is not numeric")
            else:
                if not np.isfinite(distance) or distance < 0:
                    errors.append(f"rankings row {index} distance is not finite/non-negative")
    return errors


def validate_adapter_output(
    output_dir: str | Path,
    request: AdapterRequest | None = None,
    *,
    expected_query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate status, optional arrays, and optional ranking/distance tables."""

    root = Path(output_dir).resolve()
    errors: list[str] = []
    status: dict[str, Any] = {}
    status_path = root / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"status.json is unreadable: {exc}")
    if status.get("protocol_version") != ADAPTER_PROTOCOL_VERSION:
        errors.append("status.json protocol_version is unsupported or missing")
    status_value = str(status.get("status", "failed"))
    if status_value not in {"complete", "failed"}:
        errors.append("status.json status must be 'complete' or 'failed'")
    if request is not None:
        missing_expected = [name for name in request.expected_outputs if not (root / name).exists()]
        errors.extend(f"expected output is missing: {name}" for name in missing_expected)
    query_ids = list(expected_query_ids or ())
    if not query_ids and request is not None:
        query_ids = _load_ids(request.query_ids) or []
    embeddings_path = root / "embeddings.npy"
    if embeddings_path.exists():
        try:
            embeddings = np.load(embeddings_path, allow_pickle=False)
            if embeddings.ndim != 2 or (query_ids and embeddings.shape[0] != len(query_ids)):
                errors.append("embeddings.npy must be 2-D with one row per query ID")
            if embeddings.ndim == 2 and embeddings.shape[1] < 1:
                errors.append("embeddings.npy must have a positive embedding dimension")
            if not np.issubdtype(embeddings.dtype, np.floating):
                errors.append("embeddings.npy must have a floating dtype")
            if not np.isfinite(embeddings).all():
                errors.append("embeddings.npy contains non-finite values")
        except Exception as exc:
            errors.append(f"embeddings.npy is unreadable: {exc}")
    for table_name in ("rankings", "distances"):
        table_path = root / f"{table_name}.parquet"
        if not table_path.exists():
            continue
        try:
            rows = read_parquet(table_path)
        except Exception as exc:
            errors.append(f"{table_name}.parquet is unreadable: {exc}")
            continue
        if table_name == "rankings":
            errors.extend(_validate_rank_rows(rows))
        else:
            for index, row in enumerate(rows):
                if "distance" not in row:
                    errors.append(f"distances row {index} is missing distance")
                else:
                    try:
                        value = float(row["distance"])
                    except (TypeError, ValueError):
                        errors.append(f"distances row {index} distance is not numeric")
                    else:
                        if not np.isfinite(value) or value < 0:
                            errors.append(f"distances row {index} distance is invalid")
    return {
        "valid": not errors and status_value == "complete",
        "status": status_value,
        "output_dir": str(root),
        "errors": errors,
        "protocol_version": status.get("protocol_version"),
    }


def run_external_adapter(
    command: Sequence[str],
    request: AdapterRequest,
    *,
    timeout_seconds: float | None = None,
    cwd: str | Path | None = None,
) -> AdapterRunResult:
    """Run an adapter command and always retain logs plus a validation report."""

    if not command:
        raise ExternalAdapterError("adapter command must not be empty")
    request.write()
    started = perf_counter_ns()
    timed_out = False
    returncode: int | None = None
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
    elapsed = perf_counter_ns() - started
    request.output_dir.mkdir(parents=True, exist_ok=True)
    (request.output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (request.output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    if timed_out or returncode not in (None, 0):
        status = {
            "protocol_version": ADAPTER_PROTOCOL_VERSION,
            "status": "failed",
            "error": "timeout" if timed_out else f"adapter exited with code {returncode}",
        }
        (request.output_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    validation = validate_adapter_output(request.output_dir, request)
    status_value = str(validation.get("status", "failed"))
    return AdapterRunResult(
        status_value, returncode, elapsed, request.output_dir, validation, timed_out
    )


class ExternalAdapter:
    """Small object wrapper useful to orchestration code and notebooks."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str,
        version: str = "1.0",
        timeout_seconds: float | None = None,
    ) -> None:
        self.command = tuple(map(str, command))
        self.name = str(name)
        self.version = str(version)
        self.timeout_seconds = timeout_seconds

    def run(self, request: AdapterRequest, *, cwd: str | Path | None = None) -> AdapterRunResult:
        return run_external_adapter(
            self.command, request, timeout_seconds=self.timeout_seconds, cwd=cwd
        )


validate_adapter_outputs = validate_adapter_output

__all__ = [
    "ADAPTER_OPERATIONS",
    "ADAPTER_PROTOCOL_VERSION",
    "AdapterRequest",
    "AdapterRunResult",
    "ExternalAdapter",
    "ExternalAdapterError",
    "run_external_adapter",
    "validate_adapter_output",
    "validate_adapter_outputs",
]
