"""Symmetric point-set Hausdorff distance."""

from __future__ import annotations

from .._geometry import point_distances, result, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import HausdorffConfig


class HausdorffMeasure(TrajectoryMeasure):
    """Maximum of directed point-set Hausdorff distances; order/time ignored."""

    name = "hausdorff"
    version = "1.0.0"
    capabilities = MeasureCapabilities(supports_batch=True, symmetric=True)
    config: HausdorffConfig

    @property
    def config_model(self) -> type[HausdorffConfig]:
        return HausdorffConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        distances = point_distances(points_a, points_b)
        directed_ab = float(distances.min(axis=1).max())
        directed_ba = float(distances.min(axis=0).max())
        return result(
            max(directed_ab, directed_ba),
            details={"directed_a_to_b": directed_ab, "directed_b_to_a": directed_ba},
        )


SymmetricHausdorffMeasure = HausdorffMeasure
Hausdorff = HausdorffMeasure


__all__ = ["Hausdorff", "HausdorffMeasure", "SymmetricHausdorffMeasure"]
