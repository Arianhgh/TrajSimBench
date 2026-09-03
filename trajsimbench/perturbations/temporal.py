"""Timestamp and ordering perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import Perturbation, PerturbationError, timestamp_column


class TemporalJitterPerturbation(Perturbation):
    name = "temporal_jitter"
    version = "1.0"
    units = "seconds"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "change",
        "same_underlying_movement": "small_change",
    }

    def __init__(self, *, distribution: str = "normal", repair: str = "cumulative_max") -> None:
        self.distribution = distribution
        self.repair = repair

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        scale = severity
        distribution = self.distribution
        repair = self.repair
        if isinstance(severity, Mapping):
            scale = severity.get("scale_s", severity.get("scale", 0.0))
            distribution = str(severity.get("distribution", distribution))
            repair = str(severity.get("repair", repair))
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0:
            raise PerturbationError("temporal jitter scale must be finite and non-negative seconds")
        time_col = timestamp_column(points, metadata)
        if time_col is None:
            return (
                None,
                {"scale_s": scale, "distribution": distribution, "repair": repair},
                (),
                "temporal jitter requires a timestamp column",
            )
        original_times = points[:, time_col]
        if not np.isfinite(original_times).all() or np.any(np.diff(original_times) < 0):
            return (
                None,
                {"scale_s": scale, "distribution": distribution, "repair": repair},
                (),
                "temporal jitter requires finite monotonic timestamps",
            )
        if distribution == "normal":
            noise = rng.normal(0.0, scale, size=len(points))
        elif distribution == "uniform":
            noise = rng.uniform(-scale, scale, size=len(points))
        elif distribution == "laplace":
            noise = rng.laplace(0.0, scale, size=len(points))
        else:
            raise PerturbationError(f"unsupported temporal jitter distribution: {distribution}")
        jittered = original_times + noise
        repairs = 0
        if repair in {"cumulative_max", "nondecreasing"}:
            repaired = jittered.copy()
            for index in range(1, len(repaired)):
                if repaired[index] < repaired[index - 1]:
                    repaired[index] = repaired[index - 1]
                    repairs += 1
        elif repair == "sort":
            # Sorting timestamps would detach them from observations and is
            # therefore intentionally rejected rather than silently changing
            # the trajectory's temporal correspondence.
            raise PerturbationError("temporal jitter repair='sort' is not allowed")
        else:
            raise PerturbationError(f"unsupported temporal jitter repair policy: {repair}")
        out = np.array(points, copy=True, dtype=np.float64, order="C")
        out[:, time_col] = repaired
        return (
            out,
            {
                "scale_s": scale,
                "distribution": distribution,
                "repair": repair,
                "repair_count": repairs,
                "realized_mean_abs_jitter_s": float(np.mean(np.abs(repaired - original_times))),
            },
            (),
            None,
        )


class SpeedDistortionPerturbation(Perturbation):
    name = "speed_distortion"
    version = "1.0"
    units = "time_factor"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "change",
        "same_underlying_movement": "change",
    }

    def __init__(self, *, piecewise: tuple[float, ...] | None = None) -> None:
        self.piecewise = tuple(piecewise) if piecewise is not None else None

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        factor = severity
        factors = self.piecewise
        if isinstance(severity, Mapping):
            factor = severity.get("factor", severity.get("value", 1.0))
            configured = severity.get("piecewise")
            if configured is not None:
                factors = tuple(float(v) for v in configured)
        time_col = timestamp_column(points, metadata)
        if time_col is None:
            return None, {"factor": factor}, (), "speed distortion requires a timestamp column"
        times = points[:, time_col]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            return (
                None,
                {"factor": factor},
                (),
                "speed distortion requires finite monotonic timestamps",
            )
        if factors is None:
            scalar = float(factor)
            if not np.isfinite(scalar) or scalar <= 0:
                raise PerturbationError("speed factor must be a finite positive number")
            interval_factors = np.full(max(len(times) - 1, 0), scalar, dtype=np.float64)
            factor_record: Any = scalar
        else:
            if len(factors) != max(len(times) - 1, 0):
                raise PerturbationError(
                    "piecewise speed factors must have one value per time interval"
                )
            interval_factors = np.asarray(factors, dtype=np.float64)
            if not np.isfinite(interval_factors).all() or (interval_factors <= 0).any():
                raise PerturbationError("piecewise speed factors must be finite and positive")
            factor_record = interval_factors.tolist()
        intervals = np.diff(times) * interval_factors
        out_times = np.concatenate(([times[0]], times[0] + np.cumsum(intervals)))
        out = np.array(points, copy=True, dtype=np.float64, order="C")
        out[:, time_col] = out_times
        return (
            out,
            {
                "factor": factor_record,
                "preserve_first_timestamp": True,
                "interval_count": int(len(interval_factors)),
            },
            (),
            None,
        )


class ReversalPerturbation(Perturbation):
    name = "reversal"
    version = "1.0"
    units = "ordering"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "depends",
        "temporal_dynamics": "change",
        "same_underlying_movement": "change",
        "direction_aware_movement": "change",
        "route_path_structure": "depends",
    }

    def __init__(self, *, timestamp_policy: str = "rebase") -> None:
        self.timestamp_policy = timestamp_policy

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        policy = self.timestamp_policy
        if isinstance(severity, Mapping):
            policy = str(severity.get("timestamp_policy", policy))
        if policy not in {"rebase", "omit", "reverse_durations"}:
            raise PerturbationError(
                "reversal timestamp policy must be rebase, reverse_durations, or omit"
            )
        out = np.array(points[::-1], copy=True, dtype=np.float64, order="C")
        time_col = timestamp_column(points, metadata)
        if time_col is not None:
            times = points[:, time_col]
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                return (
                    None,
                    {"timestamp_policy": policy},
                    (),
                    "reversal requires finite monotonic timestamps",
                )
            if policy in {"rebase", "reverse_durations"}:
                durations = np.diff(times)[::-1]
                out[:, time_col] = np.concatenate(([times[0]], times[0] + np.cumsum(durations)))
            else:
                out[:, time_col] = np.nan
        return (
            out,
            {
                "timestamp_policy": policy,
                "preserve_segment_durations": policy in {"rebase", "reverse_durations"},
                "preserve_first_timestamp": policy in {"rebase", "reverse_durations"},
            },
            (),
            None,
        )
