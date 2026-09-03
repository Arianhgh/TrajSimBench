"""Robustness, monotonicity, and matched hard-negative summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def robustness_curve(
    severities: Sequence[float],
    clean_values: Sequence[float] | float,
    perturbed_values: Sequence[float] | Sequence[Sequence[float]],
    *,
    mode: str = "quality",
    severity_unit: str = "normalized",
) -> list[dict[str, Any]]:
    """Create raw and normalized curve rows for quality or pairwise sensitivity."""

    axis = np.asarray(severities, dtype=float)
    if axis.ndim != 1 or len(axis) == 0 or np.any(~np.isfinite(axis)):
        raise ValueError("severities must be a non-empty finite 1D sequence")
    if np.any(np.diff(axis) < 0):
        raise ValueError("severities must be ordered")
    clean = np.asarray(clean_values, dtype=float)
    perturbed = np.asarray(perturbed_values, dtype=float)
    if perturbed.ndim == 1:
        if len(perturbed) != len(axis):
            raise ValueError("perturbed_values length must equal severities length")
        clean = np.broadcast_to(clean, perturbed.shape)
        samples = [perturbed]
        clean_samples = [clean]
    elif perturbed.ndim == 2:
        if perturbed.shape[1] != len(axis):
            raise ValueError("perturbed_values second dimension must equal severities length")
        clean = np.broadcast_to(clean, (perturbed.shape[0],)) if clean.ndim == 0 else clean
        if clean.ndim != 1 or len(clean) != perturbed.shape[0]:
            raise ValueError("clean_values must be scalar or one value per source")
        samples = list(perturbed)
        clean_samples = [np.full(len(axis), value, dtype=float) for value in clean]
    else:
        raise ValueError("perturbed_values must be 1D or 2D")
    low = float(axis.min())
    high = float(axis.max())
    normalized_axis = np.zeros_like(axis) if high == low else (axis - low) / (high - low)
    rows: list[dict[str, Any]] = []
    for source_index, (source_clean, values) in enumerate(zip(clean_samples, samples, strict=True)):
        for idx, (severity, normalized, clean_value, value) in enumerate(
            zip(axis, normalized_axis, source_clean, values, strict=True)
        ):
            if mode not in {"quality", "sensitivity"}:
                raise ValueError("mode must be 'quality' or 'sensitivity'")
            if mode == "quality":
                normalized_value = float(value / clean_value) if clean_value != 0 else float("nan")
            else:
                normalized_value = float(value - clean_value)
            rows.append(
                {
                    "source_index": source_index,
                    "severity_value": float(severity),
                    "severity_unit": severity_unit,
                    "severity_index": idx,
                    "severity_normalized": float(normalized),
                    "clean_value": float(clean_value),
                    "perturbed_value": float(value),
                    "normalized_value": normalized_value,
                    "mode": mode,
                    "valid": bool(np.isfinite(normalized_value)),
                    "reason": None if np.isfinite(normalized_value) else "clean_quality_zero",
                }
            )
    return rows


def robustness_auc(
    curve: Sequence[Mapping[str, Any]],
    *,
    axis_key: str = "severity_normalized",
    value_key: str = "normalized_value",
) -> float:
    valid = [
        row for row in curve if bool(row.get("valid", True)) and np.isfinite(float(row[value_key]))
    ]
    if len(valid) < 2:
        return float("nan")
    ordered = sorted(valid, key=lambda row: float(row[axis_key]))
    x = np.asarray([float(row[axis_key]) for row in ordered])
    y = np.asarray([float(row[value_key]) for row in ordered])
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return float(trapezoid(y, x))
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0))


def monotonicity_violation_rate(
    values_by_source: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]],
    *,
    tolerance: float = 0.0,
    expected: str = "nondecreasing",
) -> dict[str, Any]:
    """Use every ordered severity pair; return per-source and aggregate rates."""

    if tolerance < 0 or expected not in {"nondecreasing", "nonincreasing"}:
        raise ValueError("invalid tolerance or expected direction")
    items = (
        list(values_by_source.items())
        if isinstance(values_by_source, Mapping)
        else list(enumerate(values_by_source))
    )
    per_source: dict[Any, dict[str, Any]] = {}
    violations = comparisons = 0
    for source, values in items:
        arr = np.asarray(values, dtype=float)
        local_violations = local_comparisons = 0
        for i in range(len(arr) - 1):
            for j in range(i + 1, len(arr)):
                if not np.isfinite(arr[i]) or not np.isfinite(arr[j]):
                    continue
                local_comparisons += 1
                if expected == "nondecreasing":
                    violated = arr[j] < arr[i] - tolerance
                else:
                    violated = arr[j] > arr[i] + tolerance
                local_violations += int(violated)
        per_source[source] = {
            "violations": local_violations,
            "comparisons": local_comparisons,
            "rate": local_violations / local_comparisons if local_comparisons else float("nan"),
        }
        violations += local_violations
        comparisons += local_comparisons
    return {
        "rate": violations / comparisons if comparisons else float("nan"),
        "violations": violations,
        "comparisons": comparisons,
        "tolerance": tolerance,
        "expected": expected,
        "per_source": per_source,
    }


def hard_negative_gap(
    random_values: Sequence[float], hard_values: Sequence[float], *, direction: str = "quality"
) -> dict[str, Any]:
    """Matched ``delta_hard = random - hard`` with paired samples."""

    random_arr = np.asarray(random_values, dtype=float)
    hard_arr = np.asarray(hard_values, dtype=float)
    if random_arr.shape != hard_arr.shape:
        raise ValueError("random and hard values must be matched and have equal shape")
    if direction not in {"quality", "error"}:
        raise ValueError("direction must be 'quality' or 'error'")
    delta = random_arr - hard_arr
    return {
        "delta_hard": float(np.mean(delta)) if len(delta) else float("nan"),
        "paired_values": delta.tolist(),
        "sample_size": int(delta.size),
        "direction": direction,
    }


compute_robustness = robustness_curve
