"""Numerical helpers shared by classical trajectory distances."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._geometry import projected_points
from ..base import as_trajectory_view


def pair_points(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    return projected_points(as_trajectory_view(a)), projected_points(as_trajectory_view(b))


def point_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(delta, axis=2)


def resample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("resampling count must be at least one")
    if len(points) == 1 or count == 1:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total = float(cumulative[-1])
    if total == 0.0:
        return np.repeat(points[:1], count, axis=0)
    # Remove repeated arc-length positions before interpolation. Keeping the
    # first point at a position makes repeated observations deterministic.
    positions, indices = np.unique(cumulative, return_index=True)
    targets = np.linspace(0.0, total, count)
    return np.column_stack(
        [np.interp(targets, positions, points[indices, axis]) for axis in range(2)]
    )


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
