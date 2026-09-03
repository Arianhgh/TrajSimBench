"""Ranking-agreement measures with explicit tie and universe semantics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np


def _as_order(values: Sequence[Any] | Mapping[Any, float]) -> list[Any]:
    if isinstance(values, Mapping):
        return [
            key
            for key, _ in sorted(values.items(), key=lambda pair: (float(pair[1]), str(pair[0])))
        ]
    return list(values)


def average_ranks(
    values: Sequence[Any] | Mapping[Any, float], *, descending: bool = False
) -> dict[Any, float]:
    """Average 1-based ranks for scores or an already ordered sequence."""

    if isinstance(values, Mapping):
        ordered = sorted(
            values.items(), key=lambda pair: (float(pair[1]), str(pair[0])), reverse=descending
        )
        scores = [float(value) for _, value in ordered]
        keys = [key for key, _ in ordered]
        ranks: dict[Any, float] = {}
        i = 0
        while i < len(keys):
            j = i + 1
            while j < len(keys) and scores[j] == scores[i]:
                j += 1
            rank = (i + 1 + j) / 2.0
            for key in keys[i:j]:
                ranks[key] = rank
            i = j
        return ranks

    ranks = {}
    for index, item in enumerate(values, start=1):
        ranks.setdefault(item, float(index))
    return ranks


def _rank_vector(
    first: Sequence[Any] | Mapping[Any, float],
    second: Sequence[Any] | Mapping[Any, float],
    candidate_universe: Iterable[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if candidate_universe is None:
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            left = set(first)
            right = set(second)
            if left != right:
                raise ValueError("ranking agreement requires identical candidate universes")
            universe = sorted(left, key=str)
        else:
            left, right = set(first), set(second)
            if left != right:
                raise ValueError("ranking agreement requires identical candidate universes")
            universe = sorted(left, key=str)
    else:
        universe = list(candidate_universe)
    if len(set(universe)) != len(universe):
        raise ValueError("candidate_universe must contain unique IDs")
    for ranking in (first, second):
        sequence = list(ranking) if not isinstance(ranking, Mapping) else list(ranking)
        if len(sequence) != len(set(sequence)):
            raise ValueError("rankings must not contain duplicate candidate IDs")
    if isinstance(first, Mapping):
        ranks_a = average_ranks(first)
    else:
        ranks_a = average_ranks(first)
    if isinstance(second, Mapping):
        ranks_b = average_ranks(second)
    else:
        ranks_b = average_ranks(second)
    missing_a = [value for value in universe if value not in ranks_a]
    missing_b = [value for value in universe if value not in ranks_b]
    if missing_a or missing_b:
        raise ValueError("both rankings must cover candidate_universe")
    return (
        np.asarray([ranks_a[value] for value in universe], dtype=float),
        np.asarray([ranks_b[value] for value in universe], dtype=float),
        universe,
    )


def kendall_tau_b(
    first: Sequence[Any] | Mapping[Any, float],
    second: Sequence[Any] | Mapping[Any, float],
    *,
    candidate_universe: Iterable[Any] | None = None,
) -> float:
    """Kendall tau-b over a common candidate universe, including ties."""

    a, b, _ = _rank_vector(first, second, candidate_universe)
    concordant = discordant = ties_a = ties_b = ties_both = 0
    for i in range(len(a) - 1):
        da = a[i] - a[i + 1 :]
        db = b[i] - b[i + 1 :]
        concordant += int(np.sum((da * db) > 0))
        discordant += int(np.sum((da * db) < 0))
        ties_a += int(np.sum(da == 0))
        ties_b += int(np.sum(db == 0))
        ties_both += int(np.sum((da == 0) & (db == 0)))
    denominator = sqrt(
        (concordant + discordant + ties_a - ties_both)
        * (concordant + discordant + ties_b - ties_both)
    )
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((concordant - discordant) / denominator)


def spearman_rho(
    first: Sequence[Any] | Mapping[Any, float],
    second: Sequence[Any] | Mapping[Any, float],
    *,
    candidate_universe: Iterable[Any] | None = None,
) -> float:
    """Spearman rho using average ranks for tied scores."""

    a, b, _ = _rank_vector(first, second, candidate_universe)
    if len(a) < 2:
        return 1.0 if np.array_equal(a, b) else 0.0
    da = a - np.mean(a)
    db = b - np.mean(b)
    denominator = float(np.linalg.norm(da) * np.linalg.norm(db))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(da, db) / denominator)


def top_k_jaccard(first: Sequence[Any], second: Sequence[Any], k: int | None = None) -> float:
    if k is None:
        k = min(len(first), len(second))
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    left, right = set(first[:k]), set(second[:k])
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def rank_biased_overlap(
    first: Sequence[Any], second: Sequence[Any], *, persistence: float = 0.9, k: int | None = None
) -> float:
    """Finite-list Rank-Biased Overlap with extrapolation tail."""

    if not 0 <= persistence < 1:
        raise ValueError("persistence must be in [0, 1)")
    depth = max(len(first), len(second)) if k is None else int(k)
    if depth < 1:
        return 0.0
    a, b = list(first), list(second)
    total = 0.0
    for d in range(1, depth + 1):
        overlap = len(set(a[:d]) & set(b[:d])) / d
        total += (1 - persistence) * (persistence ** (d - 1)) * overlap
    tail_overlap = len(set(a[:depth]) & set(b[:depth])) / depth
    total += persistence**depth * tail_overlap
    return float(total)


def pairwise_ordering_agreement(
    first: Sequence[Any] | Mapping[Any, float],
    second: Sequence[Any] | Mapping[Any, float],
    *,
    candidate_universe: Iterable[Any] | None = None,
    tie_policy: str = "exclude",
) -> float:
    """Fraction of pairwise orderings that agree under an explicit tie policy."""

    if tie_policy not in {"exclude", "half", "disagree"}:
        raise ValueError("tie_policy must be 'exclude', 'half', or 'disagree'")
    a, b, _ = _rank_vector(first, second, candidate_universe)
    agreements = considered = 0.0
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            left = a[i] - a[j]
            right = b[i] - b[j]
            if left == 0 or right == 0:
                if tie_policy == "exclude":
                    continue
                considered += 1.0
                if tie_policy == "half":
                    agreements += 0.5
                continue
            considered += 1.0
            if left * right > 0:
                agreements += 1.0
    return float(agreements / considered) if considered else float("nan")


def compare_rankings(
    first: Sequence[Any] | Mapping[Any, float],
    second: Sequence[Any] | Mapping[Any, float],
    *,
    candidate_universe: Iterable[Any] | None = None,
    top_k: int | None = None,
    rbo_persistence: float = 0.9,
    tie_policy: str = "exclude",
) -> dict[str, float | int]:
    """Return all agreement measures while retaining comparison metadata."""

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    left = _as_order(first)
    right = _as_order(second)
    universe = list(candidate_universe) if candidate_universe is not None else None
    tau = kendall_tau_b(first, second, candidate_universe=universe)
    rho = spearman_rho(first, second, candidate_universe=universe)
    depth = top_k if top_k is not None else min(len(left), len(right))
    return {
        "kendall_tau_b": tau,
        "spearman_rho": rho,
        "top_k_jaccard": top_k_jaccard(left, right, depth),
        "rank_biased_overlap": rank_biased_overlap(
            left, right, persistence=rbo_persistence, k=depth
        ),
        "pairwise_ordering_agreement": pairwise_ordering_agreement(
            first, second, candidate_universe=universe, tie_policy=tie_policy
        ),
        "comparison_depth": int(depth),
        "candidate_count": len(universe) if universe is not None else len(set(left) | set(right)),
    }


def agreement_matrix(
    rankings: Mapping[str, Mapping[Any, Sequence[Any]]], *, metric: str = "kendall_tau_b"
) -> dict[str, dict[str, float]]:
    """Build a method-by-method matrix from per-query common-universe comparisons."""

    valid_metrics = {
        "kendall_tau_b",
        "spearman_rho",
        "top_k_jaccard",
        "rank_biased_overlap",
        "pairwise_ordering_agreement",
    }
    if metric not in valid_metrics:
        raise ValueError(f"unknown agreement metric: {metric}")
    methods = sorted(rankings)
    result = {a: {b: float("nan") for b in methods} for a in methods}
    for a in methods:
        for b in methods:
            values: list[float] = []
            for query_id in sorted(set(rankings[a]) & set(rankings[b]), key=str):
                first = rankings[a][query_id]
                second = rankings[b][query_id]
                try:
                    value = float(compare_rankings(first, second)[metric])
                except (ValueError, KeyError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            result[a][b] = float(np.mean(values)) if values else float("nan")
    return result
