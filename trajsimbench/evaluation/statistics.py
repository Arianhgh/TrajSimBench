"""Small, reproducible statistical helpers with no SciPy dependency."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    statistic: str = "mean",
    confidence: float = 0.95,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        return float("nan"), float("nan")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be in (0, 1)")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    sampled = rng.integers(0, len(arr), size=(resamples, len(arr)))
    estimates = (
        np.mean(arr[sampled], axis=1) if statistic == "mean" else np.median(arr[sampled], axis=1)
    )
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    permutations: int = 5000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired samples must be 1D arrays with equal shape")
    if len(a) == 0:
        return {
            "p_value": float("nan"),
            "effect": float("nan"),
            "sample_size": 0,
            "permutations": permutations,
        }
    if permutations < 1 or alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid permutations or alternative")
    differences = a - b
    observed = float(np.mean(differences))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(differences)))
    null = np.mean(signs * differences, axis=1)
    if alternative == "greater":
        p = (np.sum(null >= observed) + 1) / (permutations + 1)
    elif alternative == "less":
        p = (np.sum(null <= observed) + 1) / (permutations + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)
    return {
        "p_value": float(p),
        "effect": observed,
        "sample_size": len(a),
        "permutations": permutations,
        "seed": seed,
        "alternative": alternative,
    }


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and within [0, 1]")
    n = len(values)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((n - np.arange(n)) * values[order])
    result = np.empty(n, dtype=float)
    result[order] = np.minimum(adjusted_sorted, 1.0)
    return result


def summarize_samples(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(arr.tolist(), resamples=resamples, seed=seed)
    return {
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "sample_size": len(arr),
        "seed": seed,
        "resamples": resamples,
    }


bootstrap_confidence_interval = bootstrap_ci
holm_bonferroni = holm_correction
