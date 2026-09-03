"""Canonical trajectory statistics."""

from __future__ import annotations

import numpy as np


def trajectory_statistics(points: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("points must have at least two columns")
    length = (
        float(np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1).sum())
        if len(array) > 1
        else 0.0
    )
    finite_time = array[:, 2][np.isfinite(array[:, 2])] if array.shape[1] >= 3 else np.empty(0)
    return {
        "num_points": int(len(array)),
        "length_m": length,
        "start_time_s": float(finite_time[0]) if len(finite_time) else None,
        "end_time_s": float(finite_time[-1]) if len(finite_time) else None,
        "duration_s": float(finite_time[-1] - finite_time[0]) if len(finite_time) else None,
    }
