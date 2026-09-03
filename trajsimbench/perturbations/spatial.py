"""Projected-coordinate perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import Perturbation, PerturbationError, copy_with_spatial_update, spatial_columns


class IndependentGPSNoisePerturbation(Perturbation):
    name = "gps_noise"
    version = "1.0"
    units = "meters"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "preserve",
    }

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        sigma = float(
            severity.get("sigma_m", severity.get("sigma", 0.0))
            if isinstance(severity, Mapping)
            else severity
        )
        if not np.isfinite(sigma) or sigma < 0:
            raise PerturbationError(
                "GPS noise severity must be a finite non-negative sigma in meters"
            )
        x_col, y_col = spatial_columns(points, metadata)
        xy = points[:, [x_col, y_col]].copy()
        noise = rng.normal(0.0, sigma, size=(len(points), 2))
        xy += noise
        out = copy_with_spatial_update(points, xy, metadata)
        return out, {"sigma_m": sigma, "noise_model": "independent_gaussian_2d"}, (), None


class CorrelatedGPSDriftPerturbation(Perturbation):
    name = "gps_drift"
    version = "1.0"
    units = "meters"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "preserve",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "preserve",
    }

    def __init__(self, *, rho: float = 0.9) -> None:
        self.rho = float(rho)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        sigma = float(
            severity.get("sigma_m", severity.get("sigma", 0.0))
            if isinstance(severity, Mapping)
            else severity
        )
        rho = self.rho
        if not np.isfinite(sigma) or sigma < 0:
            raise PerturbationError(
                "GPS drift severity must be a finite non-negative stationary sigma"
            )
        if not np.isfinite(rho) or not -1 < rho < 1:
            raise PerturbationError("GPS drift rho must be strictly between -1 and 1")
        # sigma is stationary marginal standard deviation.  The innovation
        # standard deviation is therefore sigma*sqrt(1-rho**2), matching the
        # AR(1) parameterization in the research plan.
        innovation_sigma = sigma * np.sqrt(1.0 - rho * rho)
        errors = np.empty((len(points), 2), dtype=np.float64)
        if len(points):
            errors[0] = rng.normal(0.0, sigma, size=2)
            for index in range(1, len(points)):
                errors[index] = rho * errors[index - 1] + rng.normal(0.0, innovation_sigma, size=2)
        x_col, y_col = spatial_columns(points, metadata)
        out = copy_with_spatial_update(points, points[:, [x_col, y_col]] + errors, metadata)
        return (
            out,
            {
                "sigma_m": sigma,
                "rho": rho,
                "innovation_sigma_m": float(innovation_sigma),
                "stationary_scaling": "innovation=sigma*sqrt(1-rho^2)",
            },
            (),
            None,
        )


class SpatialQuantizationPerturbation(Perturbation):
    name = "spatial_quantization"
    version = "1.0"
    units = "meters"
    notion_expectations = {
        "geometric_shape": "small_change",
        "absolute_geographic_route": "small_change",
        "same_underlying_movement": "small_change",
    }

    def __init__(self, *, origin: tuple[float, float] | None = None) -> None:
        self.origin = origin

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        width = float(
            severity.get("grid_width_m", severity.get("width_m", 0.0))
            if isinstance(severity, Mapping)
            else severity
        )
        if not np.isfinite(width) or width <= 0:
            raise PerturbationError("quantization width must be a positive number of meters")
        x_col, y_col = spatial_columns(points, metadata)
        xy = points[:, [x_col, y_col]].copy()
        if self.origin is not None:
            origin = np.asarray(self.origin, dtype=np.float64)
        else:
            configured = metadata.get("quantization_origin_m")
            if configured is not None:
                origin = np.asarray(configured, dtype=np.float64)
            else:
                origin = np.floor(np.nanmin(xy, axis=0) / width) * width
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise PerturbationError("quantization origin must contain two finite coordinates")
        quantized = origin + np.floor((xy - origin) / width) * width
        out = copy_with_spatial_update(points, quantized, metadata)
        return (
            out,
            {
                "grid_width_m": width,
                "grid_origin_m": origin.tolist(),
                "rounding": "floor_to_origin_anchored_cell",
            },
            (),
            None,
        )


class SpatialTranslationPerturbation(Perturbation):
    name = "spatial_translation"
    version = "1.0"
    units = "meters"
    notion_expectations = {
        "geometric_shape": "preserve",
        "absolute_geographic_route": "change",
        "temporal_dynamics": "preserve",
        "same_underlying_movement": "change",
        "direction_aware_movement": "preserve",
        "route_path_structure": "preserve",
    }

    def __init__(
        self,
        *,
        bearing_rad: float | None = None,
        reject_out_of_bounds: bool = False,
    ) -> None:
        self.bearing_rad = bearing_rad
        self.reject_out_of_bounds = bool(reject_out_of_bounds)

    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ):
        magnitude = severity
        bearing = self.bearing_rad
        if isinstance(severity, Mapping):
            magnitude = float(severity.get("magnitude_m", severity.get("magnitude", 0.0)))
            bearing = severity.get("bearing_rad", bearing)
        else:
            magnitude = float(magnitude)
        if not np.isfinite(magnitude) or magnitude < 0:
            raise PerturbationError(
                "translation magnitude must be a finite non-negative distance in meters"
            )
        if bearing is None:
            bearing = float(rng.uniform(0.0, 2.0 * np.pi))
        bearing = float(bearing)
        if not np.isfinite(bearing):
            raise PerturbationError("translation bearing must be finite")
        displacement = magnitude * np.array([np.cos(bearing), np.sin(bearing)], dtype=np.float64)
        x_col, y_col = spatial_columns(points, metadata)
        xy = points[:, [x_col, y_col]] + displacement
        out = copy_with_spatial_update(points, xy, metadata)
        flags: tuple[str, ...] = ()
        bounds = metadata.get("bounds", metadata.get("bbox"))
        outside = False
        if bounds is not None and len(bounds) == 4:
            lonlat = metadata.get("geographic_columns")
            if lonlat is None and out.shape[1] >= 5:
                lonlat = (0, 1)
            if lonlat is not None:
                lon, lat = int(lonlat[0]), int(lonlat[1])
                min_lon, min_lat, max_lon, max_lat = map(float, bounds)
                outside = bool(
                    (out[:, lon] < min_lon).any()
                    or (out[:, lon] > max_lon).any()
                    or (out[:, lat] < min_lat).any()
                    or (out[:, lat] > max_lat).any()
                )
        if outside and self.reject_out_of_bounds:
            return (
                None,
                {
                    "magnitude_m": magnitude,
                    "bearing_rad": bearing,
                    "displacement_m": displacement.tolist(),
                },
                (),
                "translation leaves configured dataset bounds",
            )
        if outside:
            flags = ("outside_dataset_bounds",)
        return (
            out,
            {
                "magnitude_m": magnitude,
                "bearing_rad": bearing,
                "displacement_m": displacement.tolist(),
                "bounds_checked": bounds is not None,
            },
            flags,
            None,
        )
