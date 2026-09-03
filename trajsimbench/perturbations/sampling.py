"""Missingness, resampling, and truncation perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import (
    Perturbation,
    PerturbationError,
    spatial_columns,
    subset_points,
    timestamp_column,
)


def _round_count(value: float) -> int:
    # Explicit half-up rounding avoids Python's banker rounding changing a
    # task when a config is regenerated on another runtime.
    return int(np.floor(float(value) + 0.5))


class RandomPointLossPerturbation(Perturbation):
    name = "random_point_loss"
    version = "1.0"
    units = "retention_ratio"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "preserve",
    }

    def __init__(self, *, preserve_endpoints: bool = True) -> None:
        self.preserve_endpoints = bool(preserve_endpoints)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        retention = float(
            severity.get("retention_ratio", severity.get("ratio", 0.0))
            if isinstance(severity, Mapping)
            else severity
        )
        if not np.isfinite(retention) or not 0 < retention <= 1:
            raise PerturbationError("point-loss severity must be a retention ratio in (0, 1]")
        n = len(points)
        target = _round_count(n * retention)
        if retention == 1:
            target = n
        if target < 2:
            return (
                None,
                {"retention_ratio": retention, "target_points": target},
                (),
                "retention leaves fewer than two points",
            )
        if self.preserve_endpoints and target < 2:
            return (
                None,
                {"retention_ratio": retention, "target_points": target},
                (),
                "cannot preserve both endpoints",
            )
        if self.preserve_endpoints:
            interior_count = target - 2
            if interior_count > n - 2:
                interior_count = n - 2
            chosen = (
                rng.choice(np.arange(1, n - 1), size=interior_count, replace=False)
                if interior_count
                else np.empty(0, dtype=np.int64)
            )
            indices = np.sort(np.concatenate(([0], np.asarray(chosen, dtype=np.int64), [n - 1])))
        else:
            indices = np.sort(rng.choice(np.arange(n), size=target, replace=False))
        return (
            subset_points(points, indices),
            {
                "retention_ratio": retention,
                "target_points": int(target),
                "realized_retention_ratio": float(len(indices) / n),
                "preserve_endpoints": self.preserve_endpoints,
                "sampling": "without_replacement_then_restore_original_order",
            },
            (),
            None,
        )


class ContiguousOutagePerturbation(Perturbation):
    name = "contiguous_outage"
    version = "1.0"
    units = "point_fraction"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "preserve",
    }

    def __init__(self, *, preserve_endpoints: bool = True) -> None:
        self.preserve_endpoints = bool(preserve_endpoints)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        fraction = float(
            severity.get("fraction", severity.get("value", 0.0))
            if isinstance(severity, Mapping)
            else severity
        )
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise PerturbationError("outage severity must be a fraction strictly between 0 and 1")
        n = len(points)
        available = n - 2 if self.preserve_endpoints else n
        remove = max(1, _round_count(available * fraction))
        if remove >= available or n - remove < 2:
            return (
                None,
                {"requested_fraction": fraction, "remove_points": remove},
                (),
                "outage cannot retain a valid trajectory",
            )
        low = 1 if self.preserve_endpoints else 0
        high = n - remove - (1 if self.preserve_endpoints else 0)
        if high < low:
            return (
                None,
                {"requested_fraction": fraction, "remove_points": remove},
                (),
                "no valid interior outage location",
            )
        start = int(rng.integers(low, high + 1))
        keep = np.ones(n, dtype=bool)
        keep[start : start + remove] = False
        indices = np.flatnonzero(keep)
        return (
            subset_points(points, indices),
            {
                "requested_fraction": fraction,
                "remove_points": int(remove),
                "realized_point_fraction": float(remove / n),
                "removed_start_index": start,
                "removed_end_index_exclusive": start + remove,
                "preserve_endpoints": self.preserve_endpoints,
                "rounding": "half_up",
            },
            (),
            None,
        )


class SamplingFrequencyReductionPerturbation(Perturbation):
    name = "sampling_reduction"
    version = "1.0"
    units = "ratio_or_seconds_or_meters"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "preserve",
    }

    def __init__(self, *, mode: str = "ratio", preserve_endpoints: bool = True) -> None:
        self.mode = mode
        self.preserve_endpoints = bool(preserve_endpoints)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        mode = self.mode
        value = severity
        if isinstance(severity, Mapping):
            mode = str(severity.get("mode", mode))
            value = severity.get("value", severity.get("interval", severity.get("retention_ratio")))
        if value is None:
            raise PerturbationError("sampling reduction requires a value")
        amount = float(value)
        if not np.isfinite(amount) or amount <= 0:
            raise PerturbationError("sampling reduction value must be positive")
        time_col = timestamp_column(points, metadata)
        x_col, y_col = spatial_columns(points, metadata)
        if mode in {"ratio", "retention_ratio"}:
            if amount > 1:
                raise PerturbationError("ratio sampling reduction must be in (0, 1]")
            target = len(points) if amount == 1 else max(1, _round_count(len(points) * amount))
            if self.preserve_endpoints and target < 2:
                return (
                    None,
                    {"mode": mode, "retention_ratio": amount, "target_points": target},
                    (),
                    "ratio leaves fewer than two points",
                )
            if target < 2:
                return (
                    None,
                    {"mode": mode, "retention_ratio": amount, "target_points": target},
                    (),
                    "ratio leaves fewer than two points",
                )
            if target == len(points):
                indices = np.arange(len(points))
            elif self.preserve_endpoints:
                interior = (
                    np.linspace(1, len(points) - 2, target - 2, dtype=np.int64)
                    if target > 2
                    else np.empty(0, dtype=np.int64)
                )
                indices = np.unique(np.concatenate(([0], interior, [len(points) - 1])))
            else:
                indices = np.linspace(0, len(points) - 1, target, dtype=np.int64)
        elif mode in {"time", "temporal", "seconds"}:
            if time_col is None:
                return (
                    None,
                    {"mode": mode, "interval_s": amount},
                    (),
                    "time sampling requires timestamps",
                )
            times = points[:, time_col]
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                return (
                    None,
                    {"mode": mode, "interval_s": amount},
                    (),
                    "time sampling requires finite monotonic timestamps",
                )
            indices_list = [0]
            last_time = float(times[0])
            for index in range(1, len(points) - 1):
                if float(times[index]) - last_time >= amount:
                    indices_list.append(index)
                    last_time = float(times[index])
            if self.preserve_endpoints and indices_list[-1] != len(points) - 1:
                indices_list.append(len(points) - 1)
            indices = np.asarray(indices_list, dtype=np.int64)
        elif mode in {"spatial", "meters", "distance"}:
            xy = points[:, [x_col, y_col]]
            indices_list = [0]
            anchor = xy[0]
            for index in range(1, len(points) - 1):
                if float(np.linalg.norm(xy[index] - anchor)) >= amount:
                    indices_list.append(index)
                    anchor = xy[index]
            if self.preserve_endpoints and indices_list[-1] != len(points) - 1:
                indices_list.append(len(points) - 1)
            indices = np.asarray(indices_list, dtype=np.int64)
        else:
            raise PerturbationError(f"unsupported sampling mode: {mode}")
        indices = np.unique(indices)
        if len(indices) < 2:
            return (
                None,
                {"mode": mode, "value": amount, "realized_points": int(len(indices))},
                (),
                "sampling leaves fewer than two points",
            )
        return (
            subset_points(points, indices),
            {
                "mode": mode,
                "requested_value": amount,
                "realized_points": int(len(indices)),
                "realized_retention_ratio": float(len(indices) / len(points)),
                "first_point_anchored": True,
                "preserve_endpoints": self.preserve_endpoints,
            },
            (),
            None,
        )


class TruncationPerturbation(Perturbation):
    name = "truncation"
    version = "1.0"
    units = "fraction"
    notion_expectations = {
        "geometric_shape": "small_change",
        "absolute_geographic_route": "small_change",
        "temporal_dynamics": "small_change",
        "same_underlying_movement": "change",
    }

    def __init__(self, *, side: str = "end") -> None:
        self.side = side

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        fraction = float(severity)
        side = self.side
        if isinstance(severity, Mapping):
            fraction = float(severity.get("fraction", severity.get("value", 0.0)))
            side = str(severity.get("side", side))
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise PerturbationError("truncation fraction must be strictly between 0 and 1")
        if side not in {"start", "end", "both"}:
            raise PerturbationError("truncation side must be start, end, or both")
        total = _round_count(len(points) * fraction)
        if side == "start":
            left, right = total, 0
        elif side == "end":
            left, right = 0, total
        else:
            left = total // 2
            right = total - left
        if len(points) - left - right < 2:
            return (
                None,
                {"fraction": fraction, "side": side, "remove_start": left, "remove_end": right},
                (),
                "truncation leaves fewer than two points",
            )
        out = np.array(
            points[left : len(points) - right if right else len(points)], copy=True, order="C"
        )
        return (
            out,
            {
                "fraction": fraction,
                "side": side,
                "remove_start": int(left),
                "remove_end": int(right),
                "realized_removed_fraction": float((left + right) / len(points)),
                "rounding": "half_up",
            },
            (),
            None,
        )
