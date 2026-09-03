"""Longest common subsequence trajectory distance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._geometry import point_distances, result, timestamps, validate_pair
from ..base import DistanceResult, MeasureCapabilities, TrajectoryMeasure, TrajectoryView
from ..config import BaseMethodConfig, LCSSConfig


class LCSSMeasure(TrajectoryMeasure):
    """LCSS with spatial epsilon and optional index/time separation constraints."""

    name = "lcss"
    version = "1.0.0"
    config: LCSSConfig

    def __init__(
        self,
        config: BaseMethodConfig | Mapping[str, Any] | None = None,
        **config_values: Any,
    ) -> None:
        super().__init__(config, **config_values)
        self.capabilities = MeasureCapabilities(
            supports_batch=True,
            symmetric=True,
            requires_timestamps=(
                self.config.time_delta_s is not None or self.config.delta_mode == "time"
            ),
        )

    @property
    def config_model(self) -> type[LCSSConfig]:
        return LCSSConfig

    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        points_a, points_b = validate_pair(a, b)
        uses_time = self.config.time_delta_s is not None or self.config.delta_mode == "time"
        time_a = timestamps(a) if uses_time else None
        time_b = timestamps(b) if uses_time else None
        if uses_time and (time_a is None or time_b is None):
            raise ValueError("LCSS time delta requires finite timestamps on both trajectories")

        spatial = point_distances(points_a, points_b)
        n, m = len(points_a), len(points_b)
        lengths = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matches = spatial[i - 1, j - 1] <= self.config.epsilon
                if matches and self.config.delta is not None and self.config.delta_mode == "index":
                    matches = abs((i - 1) - (j - 1)) <= self.config.delta
                if matches and self.config.delta is not None and self.config.delta_mode == "time":
                    matches = abs(time_a[i - 1] - time_b[j - 1]) <= self.config.delta  # type: ignore[index]
                if matches and self.config.time_delta_s is not None:
                    matches = abs(time_a[i - 1] - time_b[j - 1]) <= self.config.time_delta_s  # type: ignore[index]
                if matches:
                    lengths[i, j] = lengths[i - 1, j - 1] + 1
                else:
                    lengths[i, j] = max(lengths[i - 1, j], lengths[i, j - 1])
        lcss_length = int(lengths[n, m])
        denominator = min(n, m)
        distance = max(0.0, 1.0 - lcss_length / denominator)
        return result(
            float(lcss_length),
            distance=distance,
            details={
                "lcss_length": lcss_length,
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
                "delta_mode": self.config.delta_mode,
                "time_delta_s": self.config.time_delta_s,
                "normalization_denominator": denominator,
                "dp_cells": n * m,
            },
        )


LongestCommonSubsequenceMeasure = LCSSMeasure
LCSS = LCSSMeasure


__all__ = ["LCSS", "LCSSMeasure", "LongestCommonSubsequenceMeasure"]
