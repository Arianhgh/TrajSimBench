"""Shared perturbation mechanics and duck-typed trajectory helpers."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .result import PerturbationProvenance, PerturbationResult, hash_array, hash_payload


class PerturbationError(ValueError):
    """Invalid perturbation input or configuration."""


class UnsupportedPerturbationError(PerturbationError):
    """A deliberately unavailable perturbation, such as road-network detours."""


@dataclass(frozen=True, slots=True)
class TrajectoryInput:
    trajectory_id: str
    points: np.ndarray
    metadata: Mapping[str, Any]


def trajectory_input(source: Any) -> TrajectoryInput:
    if isinstance(source, np.ndarray):
        source_id = "trajectory"
        metadata: Mapping[str, Any] = {}
        raw_points = source
    else:
        if not hasattr(source, "points"):
            raise PerturbationError("trajectory must expose a points array")
        raw_points = source.points
        source_id = str(getattr(source, "trajectory_id", getattr(source, "id", "trajectory")))
        metadata = getattr(source, "metadata", {}) or {}
    points = np.asarray(raw_points)
    if points.ndim != 2:
        raise PerturbationError("trajectory points must be a 2-D array")
    if points.shape[1] < 2:
        raise PerturbationError("trajectory points require at least two spatial columns")
    points = np.array(points, dtype=np.float64, copy=True, order="C")
    points.setflags(write=False)
    return TrajectoryInput(source_id, points, MappingProxyType(dict(metadata)))


def _metadata_columns(
    metadata: Mapping[str, Any], n_columns: int
) -> tuple[int, int, int | None, int | None]:
    """Resolve x/y and lon/lat/timestamp columns for canonical and small fixtures."""

    spatial = metadata.get("projected_columns", metadata.get("spatial_columns"))
    if isinstance(spatial, (list, tuple)) and len(spatial) >= 2:
        x_col, y_col = int(spatial[0]), int(spatial[1])
    elif n_columns >= 5:
        x_col, y_col = 2, 3
    else:
        x_col, y_col = 0, 1
    lonlat = metadata.get("geographic_columns", metadata.get("lonlat_columns"))
    lon_col: int | None
    lat_col: int | None
    if isinstance(lonlat, (list, tuple)) and len(lonlat) >= 2:
        lon_col, lat_col = int(lonlat[0]), int(lonlat[1])
    elif n_columns >= 5 and (x_col, y_col) != (0, 1):
        lon_col, lat_col = 0, 1
    else:
        lon_col = lat_col = None
    return x_col, y_col, lon_col, lat_col if lon_col is not None else None


def spatial_columns(
    points: np.ndarray, metadata: Mapping[str, Any] | None = None
) -> tuple[int, int]:
    metadata = metadata or {}
    x_col, y_col, _, _ = _metadata_columns(metadata, points.shape[1])
    if max(x_col, y_col) >= points.shape[1]:
        raise PerturbationError("projected spatial columns are outside the points array")
    return x_col, y_col


def timestamp_column(points: np.ndarray, metadata: Mapping[str, Any] | None = None) -> int | None:
    metadata = metadata or {}
    value = metadata.get("timestamp_column")
    if value is None:
        if points.shape[1] >= 5:
            return 4
        if points.shape[1] == 3 and tuple(metadata.get("projected_columns", (0, 1))) == (0, 1):
            return 2
        return None
    value = int(value)
    return value if 0 <= value < points.shape[1] else None


def geographic_columns(
    points: np.ndarray, metadata: Mapping[str, Any] | None = None
) -> tuple[int, int] | None:
    metadata = metadata or {}
    lonlat = metadata.get("geographic_columns", metadata.get("lonlat_columns"))
    if isinstance(lonlat, (list, tuple)) and len(lonlat) >= 2:
        return int(lonlat[0]), int(lonlat[1])
    if points.shape[1] >= 5:
        return 0, 1
    return None


def validate_trajectory_points(
    points: np.ndarray, metadata: Mapping[str, Any] | None = None, *, min_points: int = 1
) -> None:
    metadata = metadata or {}
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 2:
        raise PerturbationError("generated points must be a 2-D array with at least two columns")
    if array.shape[0] < min_points:
        raise PerturbationError(
            f"generated trajectory has {array.shape[0]} points; requires at least {min_points}"
        )
    x_col, y_col = spatial_columns(array, metadata)
    if not np.isfinite(array[:, [x_col, y_col]]).all():
        raise PerturbationError("generated projected coordinates contain non-finite values")
    geo = geographic_columns(array, metadata)
    if geo is not None:
        lon_col, lat_col = geo
        if not np.isfinite(array[:, [lon_col, lat_col]]).all():
            raise PerturbationError("generated geographic coordinates contain non-finite values")
        if (array[:, lon_col] < -180).any() or (array[:, lon_col] > 180).any():
            raise PerturbationError("generated longitude is outside [-180, 180]")
        if (array[:, lat_col] < -90).any() or (array[:, lat_col] > 90).any():
            raise PerturbationError("generated latitude is outside [-90, 90]")
    time_col = timestamp_column(array, metadata)
    if time_col is not None:
        times = array[:, time_col]
        finite = np.isfinite(times)
        if finite.any() and not np.all(np.diff(times[finite]) >= 0):
            raise PerturbationError("generated timestamps are not monotonic")


def polyline_length(points: np.ndarray, metadata: Mapping[str, Any] | None = None) -> float:
    x_col, y_col = spatial_columns(points, metadata)
    xy = np.asarray(points)[:, [x_col, y_col]]
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _equirectangular_update(
    points: np.ndarray,
    original: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Keep optional lon/lat columns coherent without requiring pyproj.

    Canonical projects can provide ``projected_to_geographic`` in a future
    foundation layer.  This conservative local inverse is deterministic and
    suitable for small free-space changes and fixtures.
    """

    geo = geographic_columns(points, metadata)
    if geo is None:
        return
    x_col, y_col = spatial_columns(points, metadata)
    lon_col, lat_col = geo
    earth_radius = float(metadata.get("earth_radius_m", 6_371_008.8))
    lat0 = float(np.nanmean(original[:, lat_col]))
    cos_lat = max(math.cos(math.radians(lat0)), 1e-8)
    dx = points[:, x_col] - original[:, x_col]
    dy = points[:, y_col] - original[:, y_col]
    points[:, lon_col] = original[:, lon_col] + np.degrees(dx / (earth_radius * cos_lat))
    points[:, lat_col] = original[:, lat_col] + np.degrees(dy / earth_radius)


def copy_with_spatial_update(
    original: np.ndarray,
    xy: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> np.ndarray:
    metadata = metadata or {}
    result = np.array(original, dtype=np.float64, copy=True, order="C")
    x_col, y_col = spatial_columns(result, metadata)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (len(result), 2):
        raise PerturbationError("spatial update must have shape (num_points, 2)")
    before = result.copy()
    result[:, x_col] = xy[:, 0]
    result[:, y_col] = xy[:, 1]
    _equirectangular_update(result, before, metadata)
    return result


def subset_points(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.array(
        np.asarray(points)[np.asarray(indices, dtype=np.int64)],
        dtype=np.float64,
        copy=True,
        order="C",
    )


def _mutable_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_mapping(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_mutable_mapping(item) for item in value)
    return copy.deepcopy(value)


class Perturbation(ABC):
    """Base class for pure seeded transformations."""

    name: str = "perturbation"
    version: str = "1.0"
    units: str | None = None
    notion_expectations: Mapping[str, str] = MappingProxyType({})

    def apply(
        self,
        source: Any,
        *,
        severity: Any,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> PerturbationResult:
        source_view = trajectory_input(source)
        generator = rng if rng is not None else np.random.default_rng(seed)
        if not isinstance(generator, np.random.Generator):
            raise PerturbationError("rng must be a numpy.random.Generator")
        input_hash = hash_array(source_view.points)
        initial_rng_state = copy.deepcopy(generator.bit_generator.state) if seed is None else None
        try:
            points, parameters, flags, reason = self._transform(
                source_view.points, source_view.metadata, severity, generator
            )
        except (PerturbationError, TypeError, ValueError, OverflowError) as exc:
            points, parameters, flags, reason = None, {"severity": severity}, (), str(exc)
        parameters = dict(parameters)
        if initial_rng_state is not None:
            parameters["_rng_state"] = initial_rng_state
        if reason is not None:
            variant_basis = {
                "source_id": source_view.trajectory_id,
                "transformation": self.name,
                "version": self.version,
                "severity": severity,
                "parameters": parameters,
                "seed": seed,
                "input_hash": input_hash,
                "status": "not_generated",
            }
            variant_id = f"{self.name}:rejected:{hash_payload(variant_basis)[:20]}"
            provenance = PerturbationProvenance(
                variant_id,
                source_view.trajectory_id,
                self.name,
                severity,
                self.units,
                parameters,
                seed,
                self.notion_expectations,
                self.version,
                input_hash or "",
                None,
                tuple(flags) + ("not_generated",),
            )
            return PerturbationResult(
                "not_generated", source_view.trajectory_id, None, provenance, reason
            )
        result_points = np.array(points, dtype=np.float64, copy=True, order="C")
        try:
            validate_trajectory_points(result_points, source_view.metadata)
        except PerturbationError as exc:
            variant_basis = {
                "source_id": source_view.trajectory_id,
                "transformation": self.name,
                "version": self.version,
                "severity": severity,
                "parameters": parameters,
                "seed": seed,
                "input_hash": input_hash,
                "status": "not_generated",
            }
            provenance = PerturbationProvenance(
                f"{self.name}:rejected:{hash_payload(variant_basis)[:20]}",
                source_view.trajectory_id,
                self.name,
                severity,
                self.units,
                parameters,
                seed,
                self.notion_expectations,
                self.version,
                input_hash or "",
                None,
                tuple(flags) + ("not_generated",),
            )
            return PerturbationResult(
                "not_generated", source_view.trajectory_id, None, provenance, str(exc)
            )
        output_hash = hash_array(result_points)
        variant_basis = {
            "source_id": source_view.trajectory_id,
            "transformation": self.name,
            "version": self.version,
            "severity": severity,
            "parameters": parameters,
            "seed": seed,
            "input_hash": input_hash,
            "output_hash": output_hash,
        }
        variant_id = f"{self.name}:{hash_payload(variant_basis)[:20]}"
        provenance = PerturbationProvenance(
            variant_id,
            source_view.trajectory_id,
            self.name,
            severity,
            self.units,
            parameters,
            seed,
            self.notion_expectations,
            self.version,
            input_hash or "",
            output_hash,
            tuple(flags),
        )
        return PerturbationResult("generated", variant_id, result_points, provenance)

    @abstractmethod
    def _transform(
        self,
        points: np.ndarray,
        metadata: Mapping[str, Any],
        severity: Any,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, Mapping[str, Any], tuple[str, ...], str | None]:
        raise NotImplementedError

    def regenerate(self, source: Any, provenance: PerturbationProvenance) -> PerturbationResult:
        if isinstance(provenance, Mapping):
            provenance = PerturbationProvenance.from_dict(provenance)
        if provenance.transformation != self.name or provenance.generator_version != self.version:
            raise PerturbationError("provenance does not belong to this perturbation")
        if provenance.seed is None and "_rng_state" in provenance.parameters:
            generator = np.random.default_rng()
            generator.bit_generator.state = _mutable_mapping(provenance.parameters["_rng_state"])
            result = self.apply(source, severity=provenance.severity, rng=generator)
        else:
            result = self.apply(source, severity=provenance.severity, seed=provenance.seed)
        if result.provenance.input_hash != provenance.input_hash:
            raise PerturbationError("source hash differs from provenance")
        if (
            result.provenance.output_hash != provenance.output_hash
            or result.variant_id != provenance.variant_id
        ):
            raise PerturbationError("regenerated variant differs from provenance")
        return result
