"""Per-query and macro-averaged retrieval metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _grades(
    ranked_ids: Sequence[Any], relevance: Mapping[Any, float] | Sequence[float]
) -> np.ndarray:
    if isinstance(relevance, Mapping):
        values = [float(relevance.get(candidate, 0.0)) for candidate in ranked_ids]
    else:
        values = list(relevance)
        if len(values) < len(ranked_ids):
            values.extend([0.0] * (len(ranked_ids) - len(values)))
        values = values[: len(ranked_ids)]
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("relevance grades must be finite and non-negative")
    return result


def _relevance_policy(grades: np.ndarray, policy: str) -> tuple[bool, str | None]:
    if policy not in {"skip", "zero", "raise"}:
        raise ValueError("empty relevance policy must be 'skip', 'zero', or 'raise'")
    if np.any(grades > 0):
        return True, None
    if policy == "raise":
        raise ValueError("query has no relevant candidates")
    return policy == "zero", "no_relevant_candidates"


def _policy_grades(
    relevance: Mapping[Any, float] | Sequence[float], ranked_grades: np.ndarray
) -> np.ndarray:
    if isinstance(relevance, Mapping):
        return np.asarray([float(value) for value in relevance.values()], dtype=np.float64)
    return ranked_grades


def _validate_k(k: int) -> int:
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("K must be a positive integer")
    return int(k)


def recall_at_k(
    ranked_ids: Sequence[Any],
    relevance: Mapping[Any, float] | Sequence[float],
    k: int,
    *,
    empty_policy: str = "skip",
) -> float:
    k = _validate_k(k)
    grades = _grades(ranked_ids, relevance)
    valid, _ = _relevance_policy(_policy_grades(relevance, grades), empty_policy)
    if not valid:
        return 0.0
    total = (
        float(np.sum(np.asarray(list(relevance.values()), dtype=float) > 0))
        if isinstance(relevance, Mapping)
        else float(np.sum(grades > 0))
    )
    return float(np.sum(grades[:k] > 0) / total) if total else 0.0


def precision_at_k(
    ranked_ids: Sequence[Any],
    relevance: Mapping[Any, float] | Sequence[float],
    k: int,
    *,
    empty_policy: str = "skip",
) -> float:
    k = _validate_k(k)
    grades = _grades(ranked_ids, relevance)
    valid, _ = _relevance_policy(_policy_grades(relevance, grades), empty_policy)
    if not valid:
        return 0.0
    return float(np.sum(grades[:k] > 0) / k)


def hit_rate_at_k(
    ranked_ids: Sequence[Any],
    relevance: Mapping[Any, float] | Sequence[float],
    k: int,
    *,
    empty_policy: str = "skip",
) -> float:
    k = _validate_k(k)
    grades = _grades(ranked_ids, relevance)
    valid, _ = _relevance_policy(_policy_grades(relevance, grades), empty_policy)
    if not valid:
        return 0.0
    return float(np.any(grades[:k] > 0))


def ndcg_at_k(
    ranked_ids: Sequence[Any],
    relevance: Mapping[Any, float] | Sequence[float],
    k: int,
    *,
    empty_policy: str = "skip",
) -> float:
    k = _validate_k(k)
    grades = _grades(ranked_ids, relevance)
    valid, _ = _relevance_policy(_policy_grades(relevance, grades), empty_policy)
    if not valid:
        return 0.0
    actual = grades[:k]
    discounts = np.log2(np.arange(2, len(actual) + 2))
    dcg = float(np.sum((2.0**actual - 1.0) / discounts))
    all_grades = (
        np.asarray(list(relevance.values()), dtype=float)
        if isinstance(relevance, Mapping)
        else grades
    )
    ideal = np.sort(all_grades)[::-1][:k]
    ideal_dcg = float(np.sum((2.0**ideal - 1.0) / np.log2(np.arange(2, len(ideal) + 2))))
    return float(dcg / ideal_dcg) if ideal_dcg else 0.0


def mrr(
    ranked_ids: Sequence[Any],
    relevance: Mapping[Any, float] | Sequence[float],
    *,
    empty_policy: str = "skip",
) -> float:
    grades = _grades(ranked_ids, relevance)
    valid, _ = _relevance_policy(_policy_grades(relevance, grades), empty_policy)
    if not valid:
        return 0.0
    hits = np.flatnonzero(grades > 0)
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


@dataclass(frozen=True, slots=True)
class MetricRow:
    query_id: Any
    metric: str
    value: float
    k: int | None
    valid: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "metric": self.metric,
            "value": self.value,
            "k": self.k,
            "valid": self.valid,
            "reason": self.reason,
        }


def evaluate_retrieval(
    rankings: Mapping[Any, Sequence[Any]],
    relevance: Mapping[Any, Mapping[Any, float] | Sequence[float]],
    *,
    ks: Sequence[int] = (1, 5, 10),
    empty_policy: str = "skip",
) -> list[dict[str, Any]]:
    """Compute retained per-query rows for all standard retrieval metrics."""

    validated_ks = [_validate_k(value) for value in ks]
    if len(set(validated_ks)) != len(validated_ks):
        raise ValueError("K values must be unique")
    rows: list[dict[str, Any]] = []
    for query_id, ranked_ids in rankings.items():
        if query_id not in relevance:
            raise ValueError(f"missing relevance for query {query_id!r}")
        query_relevance = relevance[query_id]
        grades = _grades(ranked_ids, query_relevance)
        policy_grades = _policy_grades(query_relevance, grades)
        valid, reason = _relevance_policy(policy_grades, empty_policy)
        functions = {
            "recall": recall_at_k,
            "precision": precision_at_k,
            "hit_rate": hit_rate_at_k,
            "ndcg": ndcg_at_k,
        }
        for k in validated_ks:
            for name, function in functions.items():
                value = (
                    float(function(ranked_ids, query_relevance, k, empty_policy=empty_policy))
                    if valid
                    else 0.0
                )
                rows.append(MetricRow(query_id, f"{name}@{k}", value, k, valid, reason).as_dict())
        value = mrr(ranked_ids, query_relevance, empty_policy=empty_policy) if valid else 0.0
        rows.append(MetricRow(query_id, "mrr", float(value), None, valid, reason).as_dict())
    return rows


def aggregate_query_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = ("metric", "k"),
    include_invalid: bool = False,
) -> list[dict[str, Any]]:
    """Macro-average query rows, retaining coverage and sample size."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_by)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        valid_values = [row for row in values if bool(row.get("valid", True))]
        selected_values = values if include_invalid else valid_values
        numeric = np.asarray([float(row["value"]) for row in selected_values], dtype=float)
        base = {name: value for name, value in zip(group_by, key, strict=True)}
        base.update(
            {
                "value": float(np.mean(numeric)) if len(numeric) else float("nan"),
                "sample_size": len(selected_values),
                "coverage": len(selected_values) / len(values) if values else 0.0,
                "aggregation_policy": "macro_query_mean",
            }
        )
        output.append(base)
    return output


recall = recall_at_k
precision = precision_at_k
hit_rate = hit_rate_at_k
ndcg = ndcg_at_k
mrr_score = mrr
