"""Parquet I/O with a deterministic JSONL fallback for minimal installs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import SCHEMA_VERSION, normalize_rows, validate_rows

FALLBACK_HEADER = "#TRAJSIMBENCH_JSONL schema_version=" + SCHEMA_VERSION


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_default(getattr(value, key)) for key in value.__dataclass_fields__}
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_fallback(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(FALLBACK_HEADER + "\n")
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, default=json_default, separators=(",", ":"))
                + "\n"
            )


def _read_fallback(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if first and not first.startswith("#TRAJSIMBENCH_JSONL"):
            try:
                rows.append(json.loads(first))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} is neither readable Parquet nor TrajSimBench JSONL"
                ) from exc
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_parquet(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    table: str | None = None,
    common: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = (
        normalize_rows(rows, table=table, common=common) if table else [dict(row) for row in rows]
    )
    if validate and table:
        report = validate_rows(row_list, table=table)
        if not report["valid"]:
            raise ValueError(f"invalid {table} rows: {'; '.join(report['errors'])}")
    temp = path.with_name(path.name + ".tmp")
    try:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            _write_fallback(temp, row_list)
        else:
            if row_list:
                arrow_table = pa.Table.from_pylist(row_list)
            else:
                columns = {
                    key: pa.array([], type=pa.string())
                    for key in ((table and ("schema_version",) + tuple()) or ("schema_version",))
                }
                arrow_table = pa.table(columns)
            metadata = dict(arrow_table.schema.metadata or {})
            metadata[b"trajsimbench.schema_version"] = SCHEMA_VERSION.encode("utf-8")
            arrow_table = arrow_table.replace_schema_metadata(metadata)
            pq.write_table(arrow_table, temp)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    temp.replace(path)
    return path


def read_parquet(path: Path, *, as_dataframe: bool = False) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        rows = _read_fallback(path)
    else:
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception:
            rows = _read_fallback(path)
    if as_dataframe:
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError("pandas is required for as_dataframe=True") from None
    return rows


def write_partitioned(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    table: str,
    partition_by: str | Sequence[str] = (),
) -> list[Path]:
    """Write deterministic table partitions, one file per partition value."""

    fields = [partition_by] if isinstance(partition_by, str) else list(partition_by)
    values = list(rows)
    if not fields:
        return [write_parquet(Path(root) / f"{table}.parquet", values, table=table)]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in values:
        groups.setdefault(tuple(row.get(field, "unknown") for field in fields), []).append(row)
    paths: list[Path] = []
    for key, group in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        directory = Path(root)
        for field, value in zip(fields, key, strict=True):
            directory /= f"{field}={value}"
        paths.append(write_parquet(directory / f"{table}.parquet", group, table=table))
    return paths


write_table = write_parquet
read_table = read_parquet
