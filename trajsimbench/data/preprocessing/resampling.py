"""Linear interpolation for optional regular temporal resampling."""

from __future__ import annotations

import numpy as np


def resample_polyline(
    points: np.ndarray, timestamps: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float64)
    source_times = np.asarray(timestamps, dtype=np.float64)
    targets = np.asarray(target_timestamps, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(source_times):
        raise ValueError("points and timestamps must have matching row counts")
    if len(source_times) == 0 or np.any(np.diff(source_times) < 0):
        raise ValueError("source timestamps must be non-empty and non-decreasing")
    if len(np.unique(source_times)) != len(source_times):
        raise ValueError("source timestamps must be unique for interpolation")
    return np.column_stack(
        [
            np.interp(targets, source_times, coordinates[:, column])
            for column in range(coordinates.shape[1])
        ]
    )
