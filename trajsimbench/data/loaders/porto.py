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
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from trajsimbench.data.checksums import sha256_file, write_checksums
from trajsimbench.data.dataset import write_canonical_dataset
from trajsimbench.data.loaders.base import BaseLoader, LoaderInspection, PreparationResult
from trajsimbench.data.projection import project_coordinates
from trajsimbench.data.schema import TrajectoryInput
from trajsimbench.data.splitting import (
    SCALE_LIMITS,
    SPLIT_ALGORITHM_VERSION,
    make_split_bundle,
    stable_id_order,
)


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


def _valid_porto_rows(
    raw_path: Path, settings: Mapping[str, Any], inspection: LoaderInspection
) -> Iterator[tuple[int, str, np.ndarray, str | None, str | None]]:
    """Yield validated raw rows without retaining the full source in memory."""

    polyline_field = str(settings.get("polyline_field", "POLYLINE"))
    trip_id_field = str(settings.get("trip_id_field", "TRIP_ID"))
    timestamp_field = settings.get("timestamp_field")
    sentinel = settings.get("missing_data_sentinel")
    minimum = int(settings.get("min_points", 2))
    maximum = settings.get("max_points")
    bbox = settings.get("bounding_box")
    with raw_path.open("r", encoding=str(settings.get("encoding", "utf-8")), newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or polyline_field not in reader.fieldnames:
            inspection.reject("missing_polyline_field")
            inspection.total_records = 1
            return
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
            timestamp = _timestamp(row.get(str(timestamp_field))) if timestamp_field else None
            semantics = str(settings.get("timestamp_semantics", "unavailable"))
            if timestamp is not None and semantics in {"start_time_plus_interval", "start_time_15s"}:
                interval = float(settings.get("sampling_interval_s", 15.0))
                points = np.column_stack(
                    (coordinates, timestamp + np.arange(len(coordinates), dtype=np.float64) * interval)
                )
            elif timestamp is not None:
                points = np.column_stack((coordinates, np.full(len(coordinates), timestamp)))
            else:
                points = coordinates
            source_id = str(row.get(trip_id_field) or row.get("TRIP_ID") or f"row-{row_number}")
            user_id = row.get(str(settings.get("user_id_field", "user_id")))
            mode = row.get(str(settings.get("mobility_mode_field", "MISSING")))
            inspection.accepted_records += 1
            yield row_number, source_id, points, str(user_id) if user_id else None, str(mode) if mode else None


def _rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{SPLIT_ALGORITHM_VERSION}:{seed}:{value}".encode()).digest()


def _write_id_array(cursor: sqlite3.Cursor, path: Path, query: str, parameters: tuple[Any, ...] = ()) -> None:
    np.save(path, np.asarray([row[0] for row in cursor.execute(query, parameters)], dtype=str), allow_pickle=False)


def _assign_partition(
    cursor: sqlite3.Cursor, *, column: str, order_by: str, counts: tuple[int, int, int]
) -> None:
    labels = ("train", "val", "test")
    boundaries = (counts[0], counts[0] + counts[1])
    rows = cursor.execute(f"SELECT row_number FROM records ORDER BY {order_by}").fetchall()
    assignments = [
        (labels[0] if index < boundaries[0] else labels[1] if index < boundaries[1] else labels[2], row[0])
        for index, row in enumerate(rows)
    ]
    cursor.executemany(f"UPDATE records SET {column} = ? WHERE row_number = ?", assignments)


def _prepare_streaming_porto(
    raw_path: Path, output_path: Path, settings: Mapping[str, Any]
) -> PreparationResult:
    """Prepare full Porto on constrained hardware without materializing all trajectories in RAM."""

    destination = output_path.resolve()
    if destination.exists():
        raise FileExistsError(f"processed dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    index_path = temporary / "records.sqlite"
    inspection = LoaderInspection(raw_path)
    try:
        connection = sqlite3.connect(index_path)
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE records (row_number INTEGER PRIMARY KEY, trajectory_id TEXT NOT NULL, "
            "source_id TEXT NOT NULL, point_count INTEGER NOT NULL, start_time REAL NOT NULL, "
            "split_rank BLOB NOT NULL, retrieval_rank BLOB NOT NULL, standard TEXT, temporal TEXT, "
            "duplicate_source INTEGER NOT NULL DEFAULT 0)"
        )
        cursor.execute("CREATE TABLE source_counts (source_id TEXT PRIMARY KEY, count INTEGER NOT NULL)")
        split_seed = int(settings.get("split_seed", 2025))
        retrieval_seed = int(settings.get("query_database_seed", 2025))
        total_points = 0
        for row_number, source_id, points, _user_id, _mode in _valid_porto_rows(
            raw_path, settings, inspection
        ):
            trajectory_id = f"porto:row-{row_number}"
            cursor.execute(
                "INSERT INTO records (row_number, trajectory_id, source_id, point_count, start_time, "
                "split_rank, retrieval_rank) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row_number,
                    trajectory_id,
                    source_id,
                    len(points),
                    float(points[0, 2]),
                    _rank(split_seed, trajectory_id),
                    _rank(retrieval_seed, trajectory_id),
                ),
            )
            cursor.execute(
                "INSERT INTO source_counts (source_id, count) VALUES (?, 1) "
                "ON CONFLICT(source_id) DO UPDATE SET count = count + 1",
                (source_id,),
            )
            total_points += len(points)
        connection.commit()
        count = int(cursor.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        if not count:
            raise ValueError(f"Porto preparation produced no valid trajectories: {inspection.rejected_by_reason}")
        duplicate_count = int(
            cursor.execute("SELECT COALESCE(SUM(count - 1), 0) FROM source_counts WHERE count > 1").fetchone()[0]
        )
        if duplicate_count:
            inspection.details["duplicate_source_ids"] = duplicate_count
            cursor.execute(
                "UPDATE records SET duplicate_source = 1 WHERE source_id IN "
                "(SELECT source_id FROM source_counts WHERE count > 1)"
            )
        train_count = int(round(count * 0.7))
        val_count = int(round(count * 0.1))
        _assign_partition(
            cursor,
            column="standard",
            order_by="split_rank, trajectory_id",
            counts=(train_count, val_count, count - train_count - val_count),
        )
        _assign_partition(
            cursor,
            column="temporal",
            order_by="start_time, row_number",
            counts=(train_count, val_count, count - train_count - val_count),
        )
        connection.commit()

        points_path = temporary / "points.npy"
        offsets_path = temporary / "offsets.npy"
        canonical_points = np.lib.format.open_memmap(
            points_path, mode="w+", dtype=np.float64, shape=(total_points, 5)
        )
        offsets = np.lib.format.open_memmap(offsets_path, mode="w+", dtype=np.int64, shape=(count + 1,))
        offsets[0] = 0
        metadata_rows: list[dict[str, Any]] = []
        metadata_path = temporary / "metadata.jsonl"
        point_offset = 0
        lon_min, lat_min = float("inf"), float("inf")
        lon_max, lat_max = float("-inf"), float("-inf")
        with metadata_path.open("w", encoding="utf-8") as metadata_handle:
            for index, (row_number, _source_id, points, _user_id, _mode) in enumerate(
                _valid_porto_rows(raw_path, settings, LoaderInspection(raw_path))
            ):
                row = cursor.execute(
                    "SELECT trajectory_id, source_id, standard, duplicate_source FROM records WHERE row_number = ?",
                    (row_number,),
                ).fetchone()
                if row is None:
                    continue
                longitude, latitude = points[:, 0], points[:, 1]
                x_m, y_m = project_coordinates(longitude, latitude, str(settings.get("projected_crs", "EPSG:32629")))
                timestamps = points[:, 2]
                canonical = np.column_stack((longitude, latitude, x_m, y_m, timestamps))
                canonical_points[point_offset : point_offset + len(canonical)] = canonical
                offsets[index + 1] = point_offset + len(canonical)
                start_time, end_time = float(timestamps[0]), float(timestamps[-1])
                length_m = float(np.linalg.norm(np.diff(canonical[:, 2:4], axis=0), axis=1).sum())
                metadata_rows.append(
                    {
                        "trajectory_idx": index,
                        "trajectory_id": row[0],
                        "dataset": "porto",
                        "source_id": row[1],
                        "user_id": None,
                        "start_time_s": start_time,
                        "end_time_s": end_time,
                        "mobility_mode": None,
                        "length_m": length_m,
                        "duration_s": end_time - start_time,
                        "num_points": len(canonical),
                        "split": row[2],
                        "crs_projected": str(settings.get("projected_crs", "EPSG:32629")),
                        "quality_flags": '["duplicate_source_id"]' if row[3] else "[]",
                    }
                )
                if len(metadata_rows) >= 10_000:
                    for metadata in metadata_rows:
                        metadata_handle.write(json.dumps(metadata, sort_keys=True) + "\n")
                    metadata_rows.clear()
                point_offset += len(canonical)
                lon_min, lon_max = min(lon_min, float(longitude.min())), max(lon_max, float(longitude.max()))
                lat_min, lat_max = min(lat_min, float(latitude.min())), max(lat_max, float(latitude.max()))
            for metadata in metadata_rows:
                metadata_handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        canonical_points.flush()
        offsets.flush()
        if point_offset != total_points:
            raise RuntimeError("streaming Porto pass disagreed with its indexed point count")

        import pandas as pd

        pd.read_json(metadata_path, lines=True).to_parquet(temporary / "metadata.parquet", index=False)
        metadata_path.unlink()
        splits_root = temporary / "splits"
        for split_name, column, order_by in (
            ("standard", "standard", "split_rank, trajectory_id"),
            ("temporal", "temporal", "start_time, row_number"),
        ):
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for partition in ("train", "val", "test"):
                _write_id_array(
                    cursor,
                    split_dir / f"{partition}.npy",
                    f"SELECT trajectory_id FROM records WHERE {column} = ? ORDER BY {order_by}",
                    (partition,),
                )
        for entry in settings.get("retrieval_scales", ()):
            if not isinstance(entry, str) or entry not in SCALE_LIMITS:
                raise ValueError("streaming Porto preparation requires named retrieval scales")
            database_count, query_count = SCALE_LIMITS[entry]
            test_count = int(cursor.execute("SELECT COUNT(*) FROM records WHERE standard = 'test'").fetchone()[0])
            if database_count + query_count > test_count:
                raise ValueError(f"retrieval scale {entry!r} exceeds the standard test partition")
            identifiers = [
                row[0]
                for row in cursor.execute(
                    "SELECT trajectory_id FROM records WHERE standard = 'test' "
                    "ORDER BY retrieval_rank, trajectory_id LIMIT ?",
                    (database_count + query_count,),
                )
            ]
            split_dir = splits_root / f"retrieval_{entry}"
            split_dir.mkdir(parents=True, exist_ok=True)
            np.save(split_dir / "database.npy", np.asarray(identifiers[:database_count], dtype=str), allow_pickle=False)
            np.save(split_dir / "query.npy", np.asarray(identifiers[database_count:], dtype=str), allow_pickle=False)
        manifest = {
            "schema_version": "1.0",
            "dataset": "porto",
            "version": str(settings.get("version", "v1")),
            "source_name": "Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015",
            "source_url": settings.get("source_url"),
            "source_license": settings.get("source_license"),
            "redistribution_policy": "raw input remains outside Git",
            "raw_checksums": {raw_path.name: sha256_file(raw_path)},
            "preprocessing_config_hash": str(
                settings.get("preprocessing_config_hash") or _preprocessing_hash(settings)
            ),
            "code_version": None,
            "projected_crs": str(settings.get("projected_crs", "EPSG:32629")),
            "projected_crs_policy": None,
            "point_columns": ["lon_deg", "lat_deg", "x_m", "y_m", "timestamp_s"],
            "point_features": {},
            "trajectory_count": count,
            "point_count": total_points,
            "bounding_box_wgs84": [lon_min, lat_min, lon_max, lat_max],
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        (temporary / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        connection.close()
        index_path.unlink()
        write_checksums(temporary)
        from trajsimbench.data.validation import validate_dataset

        report = validate_dataset(temporary, min_points=int(settings.get("min_points", 2)))
        report.raise_if_invalid()
        os.replace(temporary, destination)
        return PreparationResult(destination, inspection)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


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
        records: list[TrajectoryInput] = []
        inspection = LoaderInspection(raw_path)
        source_id_counts: dict[str, int] = {}
        for row_number, source_id, points, user_id, mode in _valid_porto_rows(
            raw_path, settings, inspection
        ):
            occurrence = source_id_counts.get(source_id, 0)
            source_id_counts[source_id] = occurrence + 1
            trajectory_id = f"porto:{source_id}" if occurrence == 0 else f"porto:{source_id}:row-{row_number}"
            if occurrence:
                inspection.details["duplicate_source_ids"] = (
                    inspection.details.get("duplicate_source_ids", 0) + 1
                )
            records.append(
                TrajectoryInput(
                    trajectory_id,
                    points,
                    source_id=source_id,
                    user_id=user_id,
                    mobility_mode=mode,
                    quality_flags=("duplicate_source_id",) if occurrence else (),
                )
            )
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
        if bool(settings.get("streaming", False)):
            result = _prepare_streaming_porto(path, Path(output_path), settings)
            self.last_inspection = result.inspection
            return result
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
