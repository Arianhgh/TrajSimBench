"""Quality/latency Pareto summaries for benchmark reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _valid(row: Mapping[str, Any]) -> bool:
    try:
        return math.isfinite(float(row["quality"])) and math.isfinite(float(row["latency_ns"]))
    except (KeyError, TypeError, ValueError):
        return False


def pareto_frontier(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep non-dominated rows, maximizing quality and minimizing latency."""
    candidates = [dict(row) for row in rows if _valid(row)]
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        quality = float(candidate["quality"])
        latency = float(candidate["latency_ns"])
        dominated = any(
            float(other["quality"]) >= quality
            and float(other["latency_ns"]) <= latency
            and (float(other["quality"]) > quality or float(other["latency_ns"]) < latency)
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (-float(row["quality"]), float(row["latency_ns"])))


def pareto_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return explicit counts alongside the retained frontier."""
    input_rows = [dict(row) for row in rows]
    frontier = pareto_frontier(input_rows)
    return {
        "input_count": len(input_rows),
        "valid_count": sum(_valid(row) for row in input_rows),
        "frontier_count": len(frontier),
        "frontier": frontier,
    }
