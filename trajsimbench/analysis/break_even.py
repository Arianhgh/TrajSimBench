"""Small helpers for comparing one-time and per-query system costs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def break_even_quantity(
    fixed_cost_a: float,
    per_query_cost_a: float,
    per_query_cost_b: float,
) -> dict[str, Any]:
    """Return the query count where system ``a`` repays its fixed cost.

    ``fixed_cost_a`` is the extra one-time cost of system ``a`` compared with
    system ``b``. A crossing exists only when ``a`` is cheaper per query.
    """
    values = (fixed_cost_a, per_query_cost_a, per_query_cost_b)
    if any(value < 0 for value in values):
        raise ValueError("costs must be non-negative")
    saving_per_query = per_query_cost_a - per_query_cost_b
    if saving_per_query <= 0:
        return {
            "break_even_queries": None,
            "finite": False,
            "fixed_cost": float(fixed_cost_a),
            "saving_per_query": float(saving_per_query),
        }
    return {
        "break_even_queries": float(fixed_cost_a / saving_per_query),
        "finite": True,
        "fixed_cost": float(fixed_cost_a),
        "saving_per_query": float(saving_per_query),
    }


def break_even_curve(
    fixed_cost_a: float,
    per_query_cost_a: float,
    per_query_cost_b: float,
    query_counts: Iterable[float],
) -> list[dict[str, float]]:
    """Return total-cost comparisons at each supplied query count."""
    result = break_even_quantity(fixed_cost_a, per_query_cost_a, per_query_cost_b)
    rows: list[dict[str, float]] = []
    for count in query_counts:
        if count < 0:
            raise ValueError("query counts must be non-negative")
        rows.append(
            {
                "query_count": float(count),
                "system_a_total": float(fixed_cost_a + (per_query_cost_a * count)),
                "system_b_total": float(per_query_cost_b * count),
                "break_even_queries": float(result["break_even_queries"])
                if result["finite"]
                else float("nan"),
            }
        )
    return rows
