"""Canonical dataset writer and read-only in-memory views."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from trajsimbench.data.checksums import read_checksums, verify_checksums, write_checksums
from trajsimbench.data.projection import project_coordinates
from trajsimbench.data.schema import (
    POINT_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    SCHEMA_VERSION,
    TrajectoryInput,
)


@dataclass(frozen=True, slots=True)
class TrajectoryView:
    """A non-owning, read-only slice of canonical points."""

    trajectory_id: str
    points: np.ndarray
    metadata: Mapping[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _quality_flags(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
    else:
        parsed = list(value) if isinstance(value, (list, tuple, set)) else [str(value)]
    return json.dumps(sorted({str(item) for item in parsed}), separators=(",", ":"))


def _coerce_input(item: TrajectoryInput | Mapping[str, Any]) -> TrajectoryInput:
    if isinstance(item, TrajectoryInput):
        return item
    value = dict(item)
    if "trajectory_id" not in value:
        raise ValueError("each trajectory input needs trajectory_id")
    points = value.pop("points", value.pop("coordinates", None))
    if points is None:
        raise ValueError("each trajectory input needs points")
    known = {"trajectory_id", "source_id", "user_id", "mobility_mode", "quality_flags", "metadata"}
    metadata = dict(value.pop("metadata", {}))
    metadata.update({key: value.pop(key) for key in list(value) if key not in known})
    return TrajectoryInput(points=np.asarray(points), metadata=metadata, **value)


def _extract_lon_lat_time(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=object)
    if array.shape[1] == 5:
        longitude = np.asarray(array[:, 0], dtype=np.float64)
        latitude = np.asarray(array[:, 1], dtype=np.float64)
        timestamp = np.asarray(array[:, 4], dtype=np.float64)
    else:
        longitude = np.asarray(array[:, 0], dtype=np.float64)
        latitude = np.asarray(array[:, 1], dtype=np.float64)
        timestamp = (
            np.asarray(array[:, 2], dtype=np.float64)
            if array.shape[1] == 3
            else np.full(array.shape[0], np.nan, dtype=np.float64)
        )
    return longitude, latitude, timestamp


def _normalize_splits(
    splits: Mapping[str, Any] | None,
    trajectory_ids: list[str],
) -> dict[str, dict[str, list[str]]]:
    if not splits:
        return {}
    partition_names = {"train", "val", "validation", "test", "query", "database"}
    keys = set(splits)
    if keys and keys.issubset(partition_names):
        source: dict[str, Any] = {"standard": splits}
    else:
        source = dict(splits)
    id_set = set(trajectory_ids)
    result: dict[str, dict[str, list[str]]] = {}
    for split_name, partition_map in source.items():
        if not isinstance(partition_map, Mapping):
            raise ValueError(f"split {split_name!r} must map partition names to IDs")
        partitions: dict[str, list[str]] = {}
        for partition, values in partition_map.items():
            if partition not in partition_names:
                raise ValueError(f"unknown split partition {partition!r}")
            ids: list[str] = []
            for value in values:
                if isinstance(value, (int, np.integer)):
                    try:
                        ids.append(trajectory_ids[int(value)])
                    except IndexError as exc:
                        raise ValueError(
                            f"split {split_name}/{partition} contains an invalid index"
                        ) from exc
                else:
                    ids.append(str(value))
            if len(ids) != len(set(ids)):
                raise ValueError(f"split {split_name}/{partition} contains duplicate IDs")
            unknown = sorted(set(ids) - id_set)
            if unknown:
                raise ValueError(
                    f"split {split_name}/{partition} contains unknown IDs: {unknown[:3]}"
                )
            partitions[partition] = ids
        result[str(split_name)] = partitions
    return result


class CanonicalDatasetWriter:
    """Write a versioned canonical dataset atomically and idempotently."""

    def write(
        self,
        path: str | Path,
        trajectories: Iterable[TrajectoryInput | Mapping[str, Any]],
        *,
        dataset: str,
        version: str = "v1",
        projected_crs: str,
        source_name: str | None = None,
        source_url: str | None = None,
        source_license: str | None = None,
        redistribution_policy: str | None = None,
        raw_checksums: Mapping[str, str] | None = None,
        preprocessing_config_hash: str | None = None,
        code_version: str | None = None,
        point_features: Mapping[str, Any] | None = None,
        feature_arrays: Mapping[str, np.ndarray] | None = None,
        splits: Mapping[str, Any] | None = None,
        acquisition_date: str | None = None,
        projected_crs_policy: str | None = None,
        created_at_utc: str | None = None,
        min_points: int = 1,
        overwrite: bool = False,
    ) -> Path:
        destination = Path(path).resolve()
        records = [_coerce_input(item) for item in trajectories]
        ids = [record.trajectory_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory IDs must be unique")
        if any(len(record.points) < min_points for record in records):
            raise ValueError(f"all trajectories must contain at least {min_points} points")
        resolved_splits = _normalize_splits(splits, ids)
        if destination.exists() and overwrite:
            raise FileExistsError(
                "overwrite is deliberately unsupported; choose a new dataset version"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            points_chunks: list[np.ndarray] = []
            offsets = [0]
            metadata_rows: list[dict[str, Any]] = []
            all_lon: list[np.ndarray] = []
            all_lat: list[np.ndarray] = []
            split_lookup: dict[str, str] = {}
            standard = resolved_splits.get("standard", {})
            for partition, partition_ids in standard.items():
                for trajectory_id in partition_ids:
                    split_lookup[trajectory_id] = partition

            for index, record in enumerate(records):
                longitude, latitude, timestamp = _extract_lon_lat_time(np.asarray(record.points))
                x_m, y_m = project_coordinates(longitude, latitude, projected_crs)
                canonical = np.column_stack((longitude, latitude, x_m, y_m, timestamp)).astype(
                    np.float64, copy=False
                )
                canonical = np.ascontiguousarray(canonical, dtype=np.float64)
                points_chunks.append(canonical)
                offsets.append(offsets[-1] + len(canonical))
                all_lon.append(longitude)
                all_lat.append(latitude)
                finite_time = timestamp[np.isfinite(timestamp)]
                start_time = float(finite_time[0]) if finite_time.size else None
                end_time = float(finite_time[-1]) if finite_time.size else None
                duration = (
                    end_time - start_time
                    if start_time is not None and end_time is not None
                    else None
                )
                length = (
                    float(np.linalg.norm(np.diff(canonical[:, 2:4], axis=0), axis=1).sum())
                    if len(canonical) > 1
                    else 0.0
                )
                row: dict[str, Any] = {
                    "trajectory_idx": index,
                    "trajectory_id": record.trajectory_id,
                    "dataset": dataset,
                    "source_id": record.source_id,
                    "user_id": record.user_id,
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "mobility_mode": record.mobility_mode,
                    "length_m": length,
                    "duration_s": duration,
                    "num_points": len(canonical),
                    "split": split_lookup.get(record.trajectory_id),
                    "crs_projected": projected_crs,
                    "quality_flags": _quality_flags(record.quality_flags),
                }
                for key, value in record.metadata.items():
                    if key not in row:
                        row[key] = _json_value(value)
                metadata_rows.append(row)

            points = np.ascontiguousarray(
                np.concatenate(points_chunks, axis=0) if points_chunks else np.empty((0, 5)),
                dtype=np.float64,
            )
            offsets_array = np.asarray(offsets, dtype=np.int64)
            np.save(temporary / "points.npy", points, allow_pickle=False)
            np.save(temporary / "offsets.npy", offsets_array, allow_pickle=False)
            metadata = pd.DataFrame(metadata_rows)
            for column in REQUIRED_METADATA_COLUMNS:
                if column not in metadata:
                    metadata[column] = None
            ordered = list(REQUIRED_METADATA_COLUMNS) + [
                column for column in metadata.columns if column not in REQUIRED_METADATA_COLUMNS
            ]
            metadata = metadata[ordered]
            metadata.to_parquet(temporary / "metadata.parquet", index=False)

            feature_declarations = dict(point_features or {})
            for name, values in (feature_arrays or {}).items():
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                    raise ValueError(f"invalid point feature name {name!r}")
                if name in {"points", "offsets", "metadata", "dataset", "checksums"}:
                    raise ValueError(
                        f"point feature name conflicts with a canonical file: {name!r}"
                    )
                array = np.asarray(values)
                if array.ndim == 0 or array.shape[0] != len(points):
                    raise ValueError(
                        f"point feature {name!r} must be row-aligned with "
                        f"points.npy ({len(points)} rows)"
                    )
                filename = f"feature_{name}.npy"
                np.save(temporary / filename, array, allow_pickle=False)
                feature_declarations[name] = {
                    "path": filename,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }

            lon_values = np.concatenate(all_lon) if all_lon else np.empty(0)
            lat_values = np.concatenate(all_lat) if all_lat else np.empty(0)
            bbox = None
            if lon_values.size:
                bbox = [
                    float(lon_values.min()),
                    float(lat_values.min()),
                    float(lon_values.max()),
                    float(lat_values.max()),
                ]
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
                "version": version,
                "source_name": source_name,
                "source_url": source_url,
                "source_license": source_license,
                "redistribution_policy": redistribution_policy,
                "raw_checksums": dict(sorted((raw_checksums or {}).items())),
                "acquisition_date": acquisition_date,
                "preprocessing_config_hash": preprocessing_config_hash,
                "code_version": code_version,
                "projected_crs": projected_crs,
                "projected_crs_policy": projected_crs_policy,
                "point_columns": list(POINT_COLUMNS),
                "point_features": feature_declarations,
                "trajectory_count": len(records),
                "point_count": int(len(points)),
                "bounding_box_wgs84": bbox,
                "created_at_utc": created_at_utc,
            }
            (temporary / "dataset.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            for split_name, partition_map in resolved_splits.items():
                split_dir = temporary / "splits" / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                for partition, partition_ids in partition_map.items():
                    np.save(
                        split_dir / f"{partition}.npy",
                        np.asarray(partition_ids, dtype=str),
                        allow_pickle=False,
                    )
            candidate_checksums = write_checksums(temporary)
            from trajsimbench.data.validation import validate_dataset

            report = validate_dataset(temporary, require_checksums=True, min_points=min_points)
            report.raise_if_invalid()
            if destination.exists():
                existing_errors = verify_checksums(destination)
                try:
                    existing_checksums = read_checksums(destination)
                except (FileNotFoundError, ValueError):
                    existing_checksums = {}
                if not existing_errors and existing_checksums == candidate_checksums:
                    shutil.rmtree(temporary, ignore_errors=True)
                    return destination
                raise FileExistsError(
                    "processed dataset already exists with different or invalid content: "
                    f"{destination}; choose a new version"
                )
            os.replace(temporary, destination)
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def write_canonical_dataset(
    path: str | Path,
    trajectories: Iterable[TrajectoryInput | Mapping[str, Any]],
    **kwargs: Any,
) -> Path:
    return CanonicalDatasetWriter().write(path, trajectories, **kwargs)


def prepare_dataset(
    dataset: str, config_path: str | Path, *, output_root: str | Path = "data/processed"
) -> Path:
    """Prepare a named dataset from a dataset YAML fragment.

    This small Python boundary is intentionally independent of the CLI. It is
    useful to callers that want the plan's ``prepare`` behavior without
    importing command parsing code.
    """

    import yaml

    from trajsimbench.data.loaders.geolife import GeoLifeLoader
    from trajsimbench.data.loaders.germany import GermanyLoader
    from trajsimbench.data.loaders.porto import PortoLoader
    from trajsimbench.data.loaders.synthetic import prepare_synthetic

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if isinstance(config.get("dataset"), dict):
        config = config["dataset"]
    if not isinstance(config, dict):
        raise ValueError("dataset config must be a YAML mapping")
    if str(config.get("name", dataset)) != dataset:
        raise ValueError(
            f"dataset argument {dataset!r} does not match config name {config.get('name')!r}"
        )
    version = str(config.get("version", "v1"))
    output = Path(output_root) / dataset / version
    if dataset == "synthetic":
        return prepare_synthetic(
            output,
            version=version,
            projected_crs=str(config.get("projected_crs", "EPSG:32633")),
            min_points=int(config.get("min_points", 1)),
        )
    raw_path = config.get("raw_path")
    if not raw_path:
        raise ValueError(f"dataset config for {dataset!r} must declare raw_path")
    if dataset == "porto":
        result = PortoLoader(config).prepare(raw_path, output)
    elif dataset == "geolife":
        result = GeoLifeLoader(config).prepare(raw_path, output)
    elif dataset == "germany":
        result = GermanyLoader().prepare(raw_path, output)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; add an explicit loader and config gate")
    return result.output_path


class TrajectoryDataset:
    """Reader for the canonical directory with read-only trajectory slices."""

    def __init__(self, path: Path, *, mmap: bool = True) -> None:
        self.path = path.resolve()
        mmap_mode: Literal["r"] | None = "r" if mmap else None
        self._points = np.load(self.path / "points.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self._offsets = np.load(self.path / "offsets.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self._metadata = pd.read_parquet(self.path / "metadata.parquet")
        self._manifest = json.loads((self.path / "dataset.json").read_text(encoding="utf-8"))
        self._id_to_index = {
            str(trajectory_id): int(index)
            for index, trajectory_id in enumerate(
                self._metadata["trajectory_id"].astype(str).tolist()
            )
        }
        self._points.flags.writeable = False
        self._offsets.flags.writeable = False

    @classmethod
    def open(cls, path: str | Path, *, mmap: bool = True) -> TrajectoryDataset:
        dataset = cls(Path(path), mmap=mmap)
        return dataset

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata.copy()

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets

    def feature(self, name: str, *, mmap: bool = True) -> np.ndarray:
        """Load one declared row-aligned optional point feature read-only."""

        declaration = self._manifest.get("point_features", {}).get(name)
        if not isinstance(declaration, Mapping) or "path" not in declaration:
            raise KeyError(f"unknown point feature: {name}")
        feature = np.load(
            self.path / str(declaration["path"]),
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
        feature.flags.writeable = False
        if len(feature) != len(self._points):
            raise ValueError(f"point feature {name!r} is not row-aligned with points.npy")
        return feature

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> TrajectoryView:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("trajectory index must be an integer")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError("trajectory index out of range")
        start, end = int(self._offsets[normalized]), int(self._offsets[normalized + 1])
        view = self._points[start:end]
        view.flags.writeable = False
        row = self._metadata.iloc[normalized].to_dict()
        row = {key: _json_value(value) for key, value in row.items()}
        if isinstance(row.get("quality_flags"), str):
            try:
                row["quality_flags"] = json.loads(row["quality_flags"])
            except json.JSONDecodeError:
                pass
        return TrajectoryView(str(row["trajectory_id"]), view, row)

    def by_id(self, trajectory_id: str) -> TrajectoryView:
        try:
            index = self._id_to_index[trajectory_id]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory_id: {trajectory_id}") from exc
        return self[index]

    def _split_files(self, split: str) -> list[Path]:
        direct = self.path / "splits" / f"{split}.npy"
        if direct.exists():
            return [direct]
        nested = self.path / "splits" / split
        if nested.is_dir():
            return sorted(nested.glob("*.npy"))
        matches = sorted(self.path.glob(f"splits/*/{split}.npy"))
        return matches

    def ids(self, split: str | None = None) -> np.ndarray:
        if split is None:
            return self._metadata["trajectory_id"].astype(str).to_numpy()
        files = self._split_files(split)
        if not files:
            raise KeyError(f"unknown split or partition: {split}")
        values: list[str] = []
        for file in files:
            raw = np.load(file, allow_pickle=False)
            for value in raw.tolist():
                if isinstance(value, (int, np.integer)):
                    values.append(str(self._metadata.iloc[int(value)]["trajectory_id"]))
                else:
                    values.append(str(value))
        return np.asarray(values, dtype=str)

    def split_names(self) -> tuple[str, ...]:
        root = self.path / "splits"
        if not root.exists():
            return ()
        return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))

    def validate(self, **kwargs: Any) -> Any:
        from trajsimbench.data.validation import validate_dataset

        return validate_dataset(self.path, **kwargs)
