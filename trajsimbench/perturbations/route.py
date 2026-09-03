"""Free-space route transformations.

Road-network detours are intentionally represented as a disabled, typed
operation.  Enabling them requires an approved map-matching source and is not
part of free-space v1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import (
    Perturbation,
    PerturbationError,
    UnsupportedPerturbationError,
    copy_with_spatial_update,
    polyline_length,
    spatial_columns,
)


def _detour_segment(
    points: np.ndarray, start: int, end: int, amplitude: float, metadata: Mapping[str, Any]
) -> np.ndarray:
    x_col, y_col = spatial_columns(points, metadata)
    xy = points[:, [x_col, y_col]].copy()
    anchor_a, anchor_b = xy[start].copy(), xy[end].copy()
    chord = anchor_b - anchor_a
    chord_length = float(np.linalg.norm(chord))
    if chord_length == 0:
        for index in range(start + 1, end + 1):
            chord = xy[index] - xy[start]
            chord_length = float(np.linalg.norm(chord))
            if chord_length:
                break
    if chord_length == 0:
        raise PerturbationError("detour requires a non-zero interior anchor chord")
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / chord_length
    span = end - start
    for offset, index in enumerate(range(start, end + 1)):
        t = offset / span
        baseline = (1.0 - t) * anchor_a + t * anchor_b
        xy[index] = baseline + normal * amplitude * np.sin(np.pi * t)
    return xy


class FreeSpaceDetourPerturbation(Perturbation):
    name = "free_space_detour"
    version = "1.0"
    units = "added_length_ratio"
    notion_expectations = {
        "geometric_shape": "small_change",
        "absolute_geographic_route": "change",
        "temporal_dynamics": "depends",
        "same_underlying_movement": "change",
        "route_path_structure": "change",
    }

    def __init__(
        self,
        *,
        anchor_fraction: tuple[float, float] = (1 / 3, 2 / 3),
        max_amplitude_factor: float = 20.0,
    ) -> None:
        self.anchor_fraction = anchor_fraction
        self.max_amplitude_factor = float(max_amplitude_factor)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        target_ratio = severity
        anchor_fraction = self.anchor_fraction
        if isinstance(severity, Mapping):
            target_ratio = severity.get("detour_ratio", severity.get("value", 0.0))
            configured = severity.get("anchor_fraction")
            if configured is not None:
                anchor_fraction = (float(configured[0]), float(configured[1]))
        target_ratio = float(target_ratio)
        if not np.isfinite(target_ratio) or target_ratio < 0:
            raise PerturbationError(
                "detour severity must be a finite non-negative added-length ratio"
            )
        if len(points) < 4:
            return (
                None,
                {"target_detour_ratio": target_ratio},
                (),
                "detour requires at least four points",
            )
        if not 0 < anchor_fraction[0] < anchor_fraction[1] < 1:
            raise PerturbationError("detour anchor fractions must satisfy 0 < start < end < 1")
        base_length = polyline_length(points, metadata)
        if base_length <= 0:
            return (
                None,
                {"target_detour_ratio": target_ratio},
                (),
                "detour requires a positive source length",
            )
        n = len(points)
        start = max(1, min(n - 3, int(np.floor(anchor_fraction[0] * (n - 1)))))
        end = min(n - 2, max(start + 1, int(np.ceil(anchor_fraction[1] * (n - 1)))))
        x_col, y_col = spatial_columns(points, metadata)
        segment_span = float(
            np.linalg.norm(points[end, [x_col, y_col]] - points[start, [x_col, y_col]])
        )
        if segment_span <= 0:
            return (
                None,
                {"target_detour_ratio": target_ratio, "anchor_indices": [start, end]},
                (),
                "detour anchor chord has zero length",
            )
        if target_ratio == 0:
            return (
                np.array(points, copy=True, dtype=np.float64, order="C"),
                {
                    "target_detour_ratio": 0.0,
                    "achieved_detour_ratio": 0.0,
                    "anchor_indices": [start, end],
                    "anchor_fractions": [float(anchor_fraction[0]), float(anchor_fraction[1])],
                    "added_length_m": 0.0,
                    "control_point_count": int(end - start + 1),
                    "max_amplitude_factor": float(self.max_amplitude_factor),
                },
                (),
                None,
            )
        # Monotone search on the normal hump amplitude.  The binary search is
        # deterministic and leaves a meaningful achieved ratio when the
        # requested ratio is not exactly representable by the sampled path.
        low, high = 0.0, max(segment_span, 1.0) * max(1.0, self.max_amplitude_factor)
        best_amp = 0.0
        best_xy = points[:, [x_col, y_col]].copy()
        best_error = float("inf")
        for _ in range(60):
            amplitude = (low + high) / 2.0
            candidate_xy = _detour_segment(points, start, end, amplitude, metadata)
            candidate = copy_with_spatial_update(points, candidate_xy, metadata)
            ratio = max(0.0, (polyline_length(candidate, metadata) - base_length) / base_length)
            error = abs(ratio - target_ratio)
            if error < best_error:
                best_amp, best_xy, best_error = amplitude, candidate_xy, error
            if ratio < target_ratio:
                low = amplitude
            else:
                high = amplitude
        out = copy_with_spatial_update(points, best_xy, metadata)
        added_length = polyline_length(out, metadata) - base_length
        achieved = max(0.0, added_length / base_length)
        flags: tuple[str, ...] = (
            () if best_error <= max(1e-9, target_ratio * 1e-3) else ("target_ratio_approximate",)
        )
        return (
            out,
            {
                "target_detour_ratio": target_ratio,
                "achieved_detour_ratio": float(achieved),
                "anchor_indices": [int(start), int(end)],
                "anchor_fractions": [float(anchor_fraction[0]), float(anchor_fraction[1])],
                "control_point_count": int(end - start + 1),
                "control_amplitude_m": float(best_amp),
                "max_amplitude_factor": float(self.max_amplitude_factor),
                "source_length_m": float(base_length),
                "added_length_m": float(added_length),
                "construction": "sinusoidal_perpendicular_hump_with_binary_search",
            },
            flags,
            None,
        )


class RoadNetworkDetourPerturbation(Perturbation):
    name = "road_network_detour"
    version = "1.0-disabled"
    units = "added_length_ratio"
    notion_expectations = {}

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        raise UnsupportedPerturbationError(
            "road-network detours are disabled until an approved "
            "map-matching source and license are configured"
        )
