"""Pure map-data helpers used by the optional Streamlit pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def trajectory_overlay(
    trajectory: Any,
    *,
    label: str | None = None,
    max_points: int = 10_000,
) -> dict[str, Any]:
    """Return a bounded WGS84/projected overlay without importing Streamlit."""

    if max_points < 1:
        raise ValueError("max_points must be positive")
    points = np.asarray(getattr(trajectory, "points", trajectory), dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("trajectory points must be a two-dimensional array")
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    if points.shape[1] >= 5:
        lon_lat = points[:, :2]
        projected = points[:, 2:4]
        timestamps = points[:, 4]
    else:
        lon_lat = points[:, :2]
        projected = points[:, :2]
        timestamps = None
    metadata = getattr(trajectory, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "trajectory_id": str(getattr(trajectory, "trajectory_id", "trajectory")),
        "label": label,
        "lon_lat": lon_lat.tolist(),
        "projected_xy": projected.tolist(),
        "timestamps_s": timestamps.tolist() if timestamps is not None else None,
        "metadata": dict(metadata),
        "point_count": len(points),
    }


def overlay_many(trajectories: Sequence[Any], *, max_points: int = 10_000) -> list[dict[str, Any]]:
    return [trajectory_overlay(item, max_points=max_points) for item in trajectories]


__all__ = ["overlay_many", "trajectory_overlay"]
