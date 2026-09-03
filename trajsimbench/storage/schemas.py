"""Schema-versioned result table contracts.

The validator is intentionally lightweight and does not require Arrow.  When
Arrow is available, :mod:`trajsimbench.storage.parquet` preserves these names
and values in actual Parquet files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "1.0"
COMMON_COLUMNS = (
    "schema_version",
    "run_id",
    "experiment_id",
    "dataset",
    "dataset_version",
    "split",
    "scale_tier",
    "method",
    "method_version",
    "method_config_hash",
    "task",
    "task_version",
    "seed",
)
TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "queries": ("query_id",),
    # The long names are the canonical v1 diagnostic schema.  The validator
    # also accepts the compact anchor/a/b aliases used by older fixtures.
    "triplets": (
        "triplet_id",
        "query_id",
        "candidate_a_id",
        "candidate_b_id",
        "notion_id",
        "expected_order",
    ),
    "variants": ("source_id", "variant_id", "perturbation", "severity_value"),
    "rankings": ("query_id", "candidate_id", "rank", "distance", "raw_score"),
    "query_metrics": ("query_id", "metric", "value", "valid"),
    "aggregate_metrics": ("metric", "value", "sample_size", "aggregation_policy"),
    "agreement": ("method_a", "method_b", "metric", "value"),
    "robustness": (
        "source_id",
        "perturbation",
        "severity_value",
        "clean_value",
        "perturbed_value",
        "normalized_value",
        "valid",
    ),
    "systems": ("stage",),
    "failures": ("stage", "error_type", "message"),
    "fingerprints": ("fingerprint_version",),
}


def _table_name(path_or_name: str) -> str:
    name = str(path_or_name).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def normalize_rows(
    rows: Iterable[Mapping[str, Any]], *, table: str, common: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    if table not in TABLE_SCHEMAS:
        raise ValueError(f"unknown result table: {table}")
    prefix = dict(common or {})
    prefix.setdefault("schema_version", SCHEMA_VERSION)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(prefix)
        row.update(dict(source))
        row.setdefault("schema_version", SCHEMA_VERSION)
        output.append(row)
    return output


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: str | None = None,
    allow_empty: bool = True,
    expected_schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    if table is not None and table not in TABLE_SCHEMAS:
        raise ValueError(f"unknown result table: {table}")
    if not rows and allow_empty:
        return {
            "valid": True,
            "row_count": 0,
            "table": table,
            "schema_version": expected_schema_version,
            "errors": [],
        }
    errors: list[str] = []
    expected = TABLE_SCHEMAS.get(table or "", ())
    for idx, row in enumerate(rows):
        if "schema_version" not in row:
            errors.append(f"row {idx}: missing required schema_version")
        elif row["schema_version"] != expected_schema_version:
            errors.append(f"row {idx}: unsupported schema_version {row.get('schema_version')!r}")
        if table == "triplets" and all(
            column in row for column in ("anchor_id", "a_id", "b_id", "expectation")
        ):
            missing = []
        else:
            missing = [column for column in expected if column not in row]
        if missing:
            errors.append(f"row {idx}: missing required columns {missing}")
        if table == "triplets" and "expected_order" in row:
            if row["expected_order"] not in {"a_closer", "b_closer", "tie", "unspecified"}:
                errors.append(f"row {idx}: invalid expected_order")
        if "rank" in row and (not isinstance(row["rank"], (int, float)) or int(row["rank"]) < 1):
            errors.append(f"row {idx}: rank must be one-based positive")
        if "valid" in row and not isinstance(row["valid"], (bool, int)):
            errors.append(f"row {idx}: valid must be boolean")
    return {
        "valid": not errors,
        "row_count": len(rows),
        "table": table,
        "schema_version": expected_schema_version,
        "errors": errors,
    }
