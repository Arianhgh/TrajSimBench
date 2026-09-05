"""Loader for user-supplied Porto-style CSV files.

The loader never downloads data. It accepts the common CSV representation with
one polyline field containing ``[[lon, lat], ...]`` and records rejection
reason counts for malformed or out-of-bounds rows.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trajsimbench.data.checksums import sha256_file
from trajsimbench.data.dataset import write_canonical_dataset
from trajsimbench.data.loaders.base import BaseLoader, LoaderInspection, PreparationResult
from trajsimbench.data.schema import TrajectoryInput
from trajsimbench.data.splitting import SCALE_LIMITS, make_split_bundle, stable_id_order


def _timestamp(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt).replace(tzinfo=UTC)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()


def _polyline(value: str, sentinel: str | None = None) -> np.ndarray | None:
    if sentinel is not None and value.strip() == sentinel:
        return None
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, (list, tuple)):
        return None
    coordinates: list[tuple[float, float]] = []
    for point in parsed:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            longitude, latitude = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        coordinates.append((longitude, latitude))
    return np.asarray(coordinates, dtype=np.float64) if coordinates else None


def _fixed_retrieval_splits(
    ids: list[str], *, settings: Mapping[str, Any]
) -> dict[str, dict[str, list[str]]]:
    """Create named, non-overlapping query/database sets from a fixed pool."""

    configured = settings.get("retrieval_scales", ())
    if not configured:
        return {}
    if not isinstance(configured, (list, tuple)):
        raise ValueError("retrieval_scales must be a list of scale names or mappings")
    seed = int(settings.get("query_database_seed", 2025))
    ordered = stable_id_order(ids, seed=seed)
    result: dict[str, dict[str, list[str]]] = {}
    for entry in configured:
        if isinstance(entry, str):
            if entry not in SCALE_LIMITS:
                raise ValueError(f"unknown retrieval scale {entry!r}")
            name = entry
            database_count, query_count = SCALE_LIMITS[entry]
        elif isinstance(entry, Mapping):
            name = str(entry.get("name", "")).strip()
            if not name:
                raise ValueError("custom retrieval scale needs a non-empty name")
            database_count = int(entry["database_count"])
            query_count = int(entry["query_count"])
        else:
            raise ValueError("each retrieval scale must be a name or mapping")
        if database_count < 1 or query_count < 1:
            raise ValueError("retrieval database_count and query_count must both be positive")
        if database_count + query_count > len(ordered):
            raise ValueError(
                f"retrieval scale {name!r} needs {database_count + query_count} test trajectories "
                f"but only {len(ordered)} are available; reduce the declared scale explicitly"
            )
        result[f"retrieval_{name}"] = {
            "database": ordered[:database_count],
            "query": ordered[database_count : database_count + query_count],
        }
    return result


def _preprocessing_hash(settings: Mapping[str, Any]) -> str:
    """Hash the effective frozen preparation settings for the dataset receipt."""

    payload = {key: value for key, value in settings.items() if key != "preprocessing_config_hash"}
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PortoLoader(BaseLoader):
    name = "porto"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.last_inspection: LoaderInspection | None = None

    def _settings(self, **kwargs: Any) -> dict[str, Any]:
        settings = dict(self.config)
        settings.update(kwargs)
        return settings

    def _read(
        self, raw_path: Path, **kwargs: Any
    ) -> tuple[list[TrajectoryInput], LoaderInspection]:
        settings = self._settings(**kwargs)
        polyline_field = str(settings.get("polyline_field", "POLYLINE"))
        trip_id_field = str(settings.get("trip_id_field", "TRIP_ID"))
        timestamp_field = settings.get("timestamp_field")
        sentinel = settings.get("missing_data_sentinel")
        minimum = int(settings.get("min_points", 2))
        maximum = settings.get("max_points")
        bbox = settings.get("bounding_box")
        records: list[TrajectoryInput] = []
        inspection = LoaderInspection(raw_path)
        with raw_path.open(
            "r", encoding=str(settings.get("encoding", "utf-8")), newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or polyline_field not in reader.fieldnames:
                inspection.reject("missing_polyline_field")
                inspection.total_records = 1
                return records, inspection
            for row_number, row in enumerate(reader, 2):
                inspection.total_records += 1
                raw_polyline = row.get(polyline_field, "") or ""
                coordinates = _polyline(raw_polyline, sentinel)
                if coordinates is None:
                    inspection.reject("malformed_or_missing_polyline")
                    continue
                if len(coordinates) < minimum:
                    inspection.reject("too_few_points")
                    continue
                if maximum is not None and len(coordinates) > int(maximum):
                    inspection.reject("too_many_points")
                    continue
                if not np.isfinite(coordinates).all():
                    inspection.reject("non_finite_coordinate")
                    continue
                if np.any((coordinates[:, 0] < -180) | (coordinates[:, 0] > 180)) or np.any(
                    (coordinates[:, 1] < -90) | (coordinates[:, 1] > 90)
                ):
                    inspection.reject("coordinate_out_of_range")
                    continue
                if bbox is not None:
                    west, south, east, north = map(float, bbox)
                    if (
                        np.any(coordinates[:, 0] < west)
                        or np.any(coordinates[:, 0] > east)
                        or np.any(coordinates[:, 1] < south)
                        or np.any(coordinates[:, 1] > north)
                    ):
                        inspection.reject("outside_configured_bounding_box")
                        continue
                timestamp = None
                if timestamp_field:
                    timestamp = _timestamp(row.get(str(timestamp_field)))
                semantics = str(settings.get("timestamp_semantics", "unavailable"))
                if timestamp is not None and semantics in {
                    "start_time_plus_interval",
                    "start_time_15s",
                }:
                    interval = float(settings.get("sampling_interval_s", 15.0))
                    times = timestamp + np.arange(len(coordinates), dtype=np.float64) * interval
                    points = np.column_stack((coordinates, times))
                elif timestamp is not None:
                    points = np.column_stack((coordinates, np.full(len(coordinates), timestamp)))
                else:
                    points = coordinates
                source_id = row.get(trip_id_field) or row.get("TRIP_ID") or f"row-{row_number}"
                trajectory_id = f"porto:{source_id}"
                user_id = row.get(str(settings.get("user_id_field", "user_id")))
                mode = row.get(str(settings.get("mobility_mode_field", "MISSING")))
                records.append(
                    TrajectoryInput(
                        trajectory_id,
                        points,
                        source_id=str(source_id),
                        user_id=str(user_id) if user_id else None,
                        mobility_mode=str(mode) if mode else None,
                    )
                )
                inspection.accepted_records += 1
        return records, inspection

    def inspect_raw(self, raw_path: str | Path, **kwargs: Any) -> LoaderInspection:
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Porto raw CSV does not exist: {path}")
        _, inspection = self._read(path, **kwargs)
        self.last_inspection = inspection
        return inspection

    def prepare(
        self, raw_path: str | Path, output_path: str | Path, **kwargs: Any
    ) -> PreparationResult:
        path = Path(raw_path).resolve()
        settings = self._settings(**kwargs)
        records, inspection = self._read(path, **kwargs)
        if not records:
            raise ValueError(
                f"Porto preparation produced no valid trajectories: {inspection.rejected_by_reason}"
            )
        ids = [record.trajectory_id for record in records]
        temporal_records = [
            {
                "trajectory_id": record.trajectory_id,
                "start_time_s": float(record.points[0, 2]),
            }
            for record in records
        ]
        split = make_split_bundle(
            ids,
            seed=int(settings.get("split_seed", 2025)),
            records=temporal_records,
            include_temporal=True,
        )
        split.update(_fixed_retrieval_splits(split["standard"]["test"], settings=settings))
        output = write_canonical_dataset(
            output_path,
            records,
            dataset="porto",
            version=str(settings.get("version", "v1")),
            projected_crs=str(settings.get("projected_crs", "EPSG:32629")),
            source_name="Porto taxi trajectory CSV supplied by the user",
            source_url=settings.get("source_url"),
            source_license=settings.get("source_license", "See the supplied source terms"),
            redistribution_policy="raw input is user-supplied and not redistributed",
            raw_checksums={path.name: sha256_file(path)},
            preprocessing_config_hash=str(
                settings.get("preprocessing_config_hash") or _preprocessing_hash(settings)
            ),
            splits=split,
            min_points=int(settings.get("min_points", 2)),
        )
        self.last_inspection = inspection
        return PreparationResult(output, inspection)

    def describe_license(self) -> Mapping[str, Any]:
        return {
            "dataset": "porto",
            "status": "user-supplied raw input required",
            "redistribution": "raw data is not included or downloaded by this loader",
            "required_action": "record the exact source citation and license in the dataset config",
        }
