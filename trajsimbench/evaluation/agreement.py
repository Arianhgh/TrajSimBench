"""Aggregate ranking-agreement evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from trajsimbench.retrieval.ranking import agreement_matrix, compare_rankings


def evaluate_agreement(
    rankings_by_method: Mapping[str, Mapping[Any, Sequence[Any]]],
    *,
    metrics: Sequence[str] = (
        "kendall_tau_b",
        "spearman_rho",
        "top_k_jaccard",
        "rank_biased_overlap",
        "pairwise_ordering_agreement",
    ),
    top_k: int | None = None,
    tie_policy: str = "exclude",
    rbo_persistence: float = 0.9,
) -> list[dict[str, Any]]:
    methods = sorted(rankings_by_method)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            queries = sorted(
                set(rankings_by_method[left]) & set(rankings_by_method[right]), key=str
            )
            per_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
            for query_id in queries:
                first = rankings_by_method[left][query_id]
                second = rankings_by_method[right][query_id]
                try:
                    comparison = compare_rankings(
                        first,
                        second,
                        top_k=top_k,
                        tie_policy=tie_policy,
                        rbo_persistence=rbo_persistence,
                    )
                except ValueError:
                    continue
                for metric in metrics:
                    if metric not in comparison:
                        raise ValueError(f"unknown agreement metric: {metric}")
                    value = float(comparison[metric])
                    if np.isfinite(value):
                        per_metric[metric].append(value)
            for metric, values in per_metric.items():
                rows.append(
                    {
                        "method_a": left,
                        "method_b": right,
                        "metric": metric,
                        "value": float(np.mean(values)) if values else float("nan"),
                        "sample_size": len(values),
                        "comparison_depth": top_k,
                    }
                )
    return rows


def agreement_distance(value: float, *, metric: str = "kendall_tau_b") -> float:
    """Map agreement to a clustering distance, handling negative values."""

    if not np.isfinite(value):
        return float("nan")
    if metric in {"kendall_tau_b", "spearman_rho", "pairwise_ordering_agreement"}:
        if metric == "pairwise_ordering_agreement":
            return float(1.0 - value)
        return float((1.0 - value) / 2.0)
    return float(1.0 - value)


def build_agreement_matrix(
    rankings_by_method: Mapping[str, Mapping[Any, Sequence[Any]]], *, metric: str = "kendall_tau_b"
) -> dict[str, dict[str, float]]:
    return agreement_matrix(rankings_by_method, metric=metric)


compute_agreement = evaluate_agreement
