"""Bounded interactive lookups; pages never recompute benchmark measures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .results import read_result_table


def pair_detail(run_dir: str | Path, first_id: Any, second_id: Any) -> dict[str, Any]:
    rows = read_result_table(run_dir, "rankings")
    relevant = [
        row
        for row in rows
        if row.get("candidate_id") in {first_id, second_id}
        or row.get("query_id") in {first_id, second_id}
    ]
    return {
        "first_id": first_id,
        "second_id": second_id,
        "rankings": relevant,
        "count": len(relevant),
    }


def retrieval_disagreement(
    run_dir: str | Path,
    query_id: Any,
    method_a: str | None = None,
    method_b: str | None = None,
    *,
    limit: int = 100,
) -> dict[str, Any]:
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be in [1, 1000]")
    rows = [
        row for row in read_result_table(run_dir, "rankings") if row.get("query_id") == query_id
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: int(item.get("rank", 0))):
        by_method.setdefault(str(row.get("method", "unknown")), []).append(row)
    if method_a is None or method_b is None:
        names = sorted(by_method)
        method_a, method_b = (names + [None, None])[:2]
    left = by_method.get(str(method_a), [])[:limit] if method_a is not None else []
    right = by_method.get(str(method_b), [])[:limit] if method_b is not None else []
    left_ids, right_ids = (
        {row.get("candidate_id") for row in left},
        {row.get("candidate_id") for row in right},
    )
    return {
        "query_id": query_id,
        "method_a": method_a,
        "method_b": method_b,
        "left": left,
        "right": right,
        "shared": sorted(left_ids & right_ids, key=str),
        "unique_a": sorted(left_ids - right_ids, key=str),
        "unique_b": sorted(right_ids - left_ids, key=str),
    }
