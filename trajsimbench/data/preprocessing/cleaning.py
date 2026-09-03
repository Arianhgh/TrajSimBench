"""Deterministic cleaning operations with counted outcomes."""

from __future__ import annotations

import numpy as np


def deduplicate_consecutive(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if len(array) < 2:
        return array.copy()
    keep = np.ones(len(array), dtype=bool)
    keep[1:] = np.any(array[1:] != array[:-1], axis=1)
    return array[keep].copy()


def clean_points(
    points: np.ndarray,
    *,
    deduplicate: bool = True,
    drop_nonfinite: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("points must be a two-dimensional array with at least two columns")
    report = {"input_points": len(array), "dropped_nonfinite": 0, "deduplicated": 0}
    if drop_nonfinite:
        finite = np.isfinite(array[:, :2]).all(axis=1)
        report["dropped_nonfinite"] = int((~finite).sum())
        array = array[finite]
    if deduplicate:
        before = len(array)
        array = deduplicate_consecutive(array)
        report["deduplicated"] = before - len(array)
    report["output_points"] = len(array)
    return array, report
