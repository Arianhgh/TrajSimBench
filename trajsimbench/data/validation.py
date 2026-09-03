"""Actionable validation for canonical processed datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS

from trajsimbench.data.checksums import verify_checksums
from trajsimbench.data.projection import project_coordinates
from trajsimbench.data.schema import POINT_COLUMNS, REQUIRED_METADATA_COLUMNS, SCHEMA_VERSION


class DatasetValidationError(ValueError):
    """Raised when a canonical dataset violates one or more invariants."""


@dataclass(slots=True)
class ValidationReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            detail = "\n".join(f"- {error}" for error in self.errors)
            raise DatasetValidationError(f"invalid canonical dataset {self.path}:\n{detail}")


def _load_array(path: Path, report: ValidationReport, name: str) -> np.ndarray | None:
    try:
        return np.load(path / name, allow_pickle=False)
    except Exception as exc:
        report.errors.append(f"cannot read {name}: {exc}")
        return None


def _split_values(file: Path) -> list[str]:
    values = np.load(file, allow_pickle=False).tolist()
    return [str(value) for value in values]


def _validate_splits(
    path: Path, ids: set[str], metadata: pd.DataFrame, report: ValidationReport
) -> None:
    root = path / "splits"
    if not root.exists():
        return
    exclusive = {"train", "val", "validation", "test"}
    for split_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        seen: dict[str, str] = {}
        partition_files = sorted(split_dir.glob("*.npy"))
        for file in partition_files:
            try:
                values = _split_values(file)
            except Exception as exc:
                report.errors.append(f"cannot read split file {file.relative_to(path)}: {exc}")
                continue
            partition = file.stem
            if len(values) != len(set(values)):
                report.errors.append(f"split {split_dir.name}/{partition} contains duplicate IDs")
            unknown = sorted(set(values) - ids)
            if unknown:
                report.errors.append(
                    f"split {split_dir.name}/{partition} contains unknown IDs, e.g. {unknown[0]!r}"
                )
            for value in values:
                if partition in exclusive and value in seen and seen[value] in exclusive:
                    report.errors.append(
                        f"trajectory {value!r} appears in mutually exclusive partitions "
                        f"{seen[value]!r} and {partition!r} of split {split_dir.name!r}"
                    )
                seen[value] = partition
        if "user" in split_dir.name.lower() or "held" in split_dir.name.lower():
            user_by_id = dict(
                zip(
                    metadata["trajectory_id"].astype(str),
                    metadata["user_id"].astype(str),
                    strict=True,
                )
            )
            partition_users: dict[str, set[str]] = {}
            for file in partition_files:
                if file.stem not in exclusive:
                    continue
                users = {user_by_id[value] for value in _split_values(file) if value in user_by_id}
                partition_users[file.stem] = users
            partitions = list(partition_users)
            for index, left in enumerate(partitions):
                for right in partitions[index + 1 :]:
                    overlap = partition_users[left].intersection(partition_users[right])
                    if overlap:
                        report.errors.append(
                            f"user-held-out split {split_dir.name!r} overlaps users between "
                            f"{left} and {right}: {sorted(overlap)[:3]}"
                        )


def validate_dataset(
    path: str | Path,
    *,
    min_points: int = 1,
    require_checksums: bool = True,
    allow_nonmonotonic_timestamps: bool = False,
    projection_tolerance_m: float = 0.05,
    sample_projection_points: int = 10_000,
) -> ValidationReport:
    """Validate all structural, spatial, temporal, split, and checksum rules."""

    dataset_path = Path(path).resolve()
    report = ValidationReport(dataset_path)
    if not dataset_path.is_dir():
        report.errors.append(f"dataset directory does not exist: {dataset_path}")
        return report
    points = _load_array(dataset_path, report, "points.npy")
    offsets = _load_array(dataset_path, report, "offsets.npy")
    try:
        metadata = pd.read_parquet(dataset_path / "metadata.parquet")
    except Exception as exc:
        metadata = pd.DataFrame()
        report.errors.append(f"cannot read metadata.parquet: {exc}")
    try:
        manifest = json.loads((dataset_path / "dataset.json").read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        report.errors.append(f"cannot read dataset.json: {exc}")

    if points is not None:
        if points.ndim != 2 or points.shape[1] != len(POINT_COLUMNS):
            report.errors.append(f"points.npy must have shape [point_count, 5], got {points.shape}")
        if points.dtype != np.float64:
            report.errors.append(f"points.npy must have dtype float64, got {points.dtype}")
        if not points.flags.c_contiguous:
            report.errors.append("points.npy must be C-contiguous")
        if points.ndim == 2 and points.shape[1] >= 2:
            lon, lat = points[:, 0], points[:, 1]
            if not np.isfinite(lon).all() or not np.isfinite(lat).all():
                report.errors.append("longitude/latitude columns contain non-finite values")
            if np.any((lon < -180) | (lon > 180)):
                report.errors.append("longitude is outside [-180, 180]")
            if np.any((lat < -90) | (lat > 90)):
                report.errors.append("latitude is outside [-90, 90]")
            if points.shape[1] >= 4 and (not np.isfinite(points[:, 2:4]).all()):
                report.errors.append("projected x_m/y_m columns contain non-finite values")

    if offsets is not None:
        if offsets.ndim != 1:
            report.errors.append("offsets.npy must be one-dimensional")
        if offsets.dtype != np.int64:
            report.errors.append(f"offsets.npy must have dtype int64, got {offsets.dtype}")
        if offsets.ndim == 1:
            if len(offsets) != len(metadata) + 1:
                report.errors.append(
                    "offsets length must equal metadata rows + 1 "
                    f"({len(metadata) + 1}), got {len(offsets)}"
                )
            if len(offsets) and offsets[0] != 0:
                report.errors.append("offsets.npy must begin at zero")
            if np.any(np.diff(offsets) < 0):
                report.errors.append("offsets.npy must be monotonically nondecreasing")
            if points is not None and len(offsets) and offsets[-1] != len(points):
                report.errors.append(
                    f"offsets last value must equal point count {len(points)}, got {offsets[-1]}"
                )

    missing_columns = sorted(set(REQUIRED_METADATA_COLUMNS) - set(metadata.columns))
    if missing_columns:
        report.errors.append(
            f"metadata.parquet is missing required columns: {', '.join(missing_columns)}"
        )
    if not metadata.empty:
        ids = metadata["trajectory_id"].astype(str)
        if ids.duplicated().any():
            report.errors.append("trajectory_id values must be unique")
        expected_indices = np.arange(len(metadata), dtype=np.int64)
        try:
            actual_indices = metadata["trajectory_idx"].to_numpy(dtype=np.int64)
            if not np.array_equal(actual_indices, expected_indices):
                report.errors.append(
                    "trajectory_idx must be contiguous and match metadata row order"
                )
        except (TypeError, ValueError):
            report.errors.append("trajectory_idx must contain integer values")
        if offsets is not None and offsets.ndim == 1 and len(offsets) == len(metadata) + 1:
            spans = np.diff(offsets)
            try:
                num_points = metadata["num_points"].to_numpy(dtype=np.int64)
                if not np.array_equal(spans, num_points):
                    report.errors.append("metadata.num_points must equal each offsets span")
                if np.any(num_points < min_points):
                    report.errors.append(f"trajectories must contain at least {min_points} points")
            except (TypeError, ValueError):
                report.errors.append("metadata.num_points must contain integer values")
        for field_name in ("length_m", "duration_s"):
            values = pd.to_numeric(metadata[field_name], errors="coerce")
            if values.notna().any() and (values.dropna() < 0).any():
                report.errors.append(f"metadata.{field_name} must be non-negative")
        if (
            offsets is not None
            and points is not None
            and offsets.ndim == 1
            and len(offsets) == len(metadata) + 1
        ):
            for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                times = points[int(start) : int(end), 4]
                finite = times[np.isfinite(times)]
                if (
                    finite.size > 1
                    and np.any(np.diff(finite) < 0)
                    and not allow_nonmonotonic_timestamps
                ):
                    report.errors.append(
                        f"timestamps are non-monotonic in trajectory {ids.iloc[index]!r}"
                    )
        if points is not None and len(points):
            crs_values = metadata["crs_projected"].dropna().astype(str).unique().tolist()
            if len(crs_values) != 1:
                report.errors.append("all trajectories must declare one resolved projected CRS")
            else:
                try:
                    CRS.from_user_input(crs_values[0])
                    sample = np.linspace(
                        0, len(points) - 1, min(len(points), sample_projection_points), dtype=int
                    )
                    x, y = project_coordinates(points[sample, 0], points[sample, 1], crs_values[0])
                    discrepancy = np.maximum(
                        np.abs(x - points[sample, 2]), np.abs(y - points[sample, 3])
                    )
                    if np.nanmax(discrepancy) > projection_tolerance_m:
                        report.errors.append(
                            f"projected coordinates disagree with {crs_values[0]} by up to "
                            f"{float(np.nanmax(discrepancy)):.3g} m "
                            f"(tolerance {projection_tolerance_m} m)"
                        )
                except Exception as exc:
                    report.errors.append(f"cannot validate projected CRS: {exc}")
        _validate_splits(dataset_path, set(ids), metadata, report)
    elif offsets is not None and len(offsets) != 1:
        report.errors.append("empty metadata requires offsets.npy to contain exactly one zero")

    if manifest:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            report.errors.append(
                "unsupported dataset schema_version "
                f"{manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        if points is not None and manifest.get("point_count") != len(points):
            report.errors.append("dataset.json point_count does not match points.npy")
        if manifest.get("trajectory_count") != len(metadata):
            report.errors.append("dataset.json trajectory_count does not match metadata.parquet")
        if manifest.get("point_columns") not in (None, list(POINT_COLUMNS)):
            report.errors.append("dataset.json point_columns do not match the canonical schema")

    if require_checksums:
        report.errors.extend(verify_checksums(dataset_path))
    return report


def validate_processed_dataset(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Dictionary compatibility wrapper for the CLI/artifact boundary."""

    report = validate_dataset(path, **kwargs)
    return {
        "valid": report.ok,
        "path": str(report.path),
        "errors": list(report.errors),
        "warnings": list(report.warnings),
    }
