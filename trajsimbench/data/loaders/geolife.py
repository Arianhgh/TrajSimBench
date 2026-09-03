"""Deterministic parser for the public GeoLife directory layout."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trajsimbench.data.checksums import sha256_file
from trajsimbench.data.dataset import write_canonical_dataset
from trajsimbench.data.loaders.base import BaseLoader, LoaderInspection, PreparationResult
from trajsimbench.data.schema import TrajectoryInput
from trajsimbench.data.splitting import make_split_bundle


def _parse_time(date_text: str, time_text: str) -> float | None:
    try:
        value = datetime.strptime(f"{date_text.strip()} {time_text.strip()}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return value.replace(tzinfo=UTC).timestamp()


def _label_index(path: Path) -> dict[str, str]:
    """Read optional labels.txt into a conservative timestamp-to-mode index."""

    labels_path = path / "labels.txt"
    if not labels_path.exists():
        return {}
    result: dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle, delimiter=" "):
            fields = [field for field in row if field]
            if len(fields) >= 3:
                result[fields[0]] = fields[-1]
    return result


class GeoLifeLoader(BaseLoader):
    name = "geolife"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.last_inspection: LoaderInspection | None = None

    def _files(self, root: Path) -> list[Path]:
        return sorted(
            root.rglob("*.plt"), key=lambda path: path.relative_to(root).as_posix().lower()
        )

    def _read_file(
        self, path: Path, root: Path, settings: Mapping[str, Any], inspection: LoaderInspection
    ) -> TrajectoryInput | None:
        relative_parts = path.relative_to(root).parts
        user_id = relative_parts[0] if len(relative_parts) > 1 else path.parent.name
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            inspection.reject("unreadable_file")
            return None
        rows: list[tuple[float, float, float]] = []
        previous: tuple[float, float, float] | None = None
        for line in lines[6:]:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                if line.strip():
                    inspection.reject("malformed_point_row")
                continue
            try:
                latitude, longitude = float(fields[0]), float(fields[1])
            except ValueError:
                inspection.reject("malformed_coordinate")
                continue
            timestamp = _parse_time(fields[4], fields[5])
            if timestamp is None:
                inspection.reject("malformed_timestamp")
                continue
            point = (longitude, latitude, timestamp)
            if (
                str(settings.get("deduplicate_policy", "drop_consecutive")) == "drop_consecutive"
                and point == previous
            ):
                inspection.details["deduplicated_points"] = (
                    inspection.details.get("deduplicated_points", 0) + 1
                )
                continue
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                inspection.reject("coordinate_out_of_range")
                continue
            rows.append(point)
            previous = point
        minimum = int(settings.get("min_points", 2))
        if len(rows) < minimum:
            inspection.reject("too_few_points")
            return None
        coordinates = np.asarray(rows, dtype=np.float64)
        mode = None
        labels = _label_index(
            path.parent.parent if path.parent.name.lower() == "trajectory" else path.parent
        )
        if labels:
            mode = labels.get(path.stem)
        relative_id = path.relative_to(root).with_suffix("").as_posix().replace("/", ":")
        trajectory_id = f"geolife:{user_id}:{relative_id}"
        inspection.accepted_records += 1
        return TrajectoryInput(
            trajectory_id,
            coordinates,
            source_id=path.relative_to(root).as_posix(),
            user_id=str(user_id),
            mobility_mode=mode,
        )

    def _read(
        self, raw_path: Path, **kwargs: Any
    ) -> tuple[list[TrajectoryInput], LoaderInspection]:
        settings = dict(self.config)
        settings.update(kwargs)
        files = self._files(raw_path)
        inspection = LoaderInspection(raw_path, total_records=len(files))
        records = [
            record
            for file in files
            if (record := self._read_file(file, raw_path, settings, inspection)) is not None
        ]
        return records, inspection

    def inspect_raw(self, raw_path: str | Path, **kwargs: Any) -> LoaderInspection:
        root = Path(raw_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"GeoLife raw directory does not exist: {root}")
        _, inspection = self._read(root, **kwargs)
        self.last_inspection = inspection
        return inspection

    def prepare(
        self, raw_path: str | Path, output_path: str | Path, **kwargs: Any
    ) -> PreparationResult:
        root = Path(raw_path).resolve()
        settings = dict(self.config)
        settings.update(kwargs)
        records, inspection = self._read(root, **kwargs)
        if not records:
            raise ValueError(
                "GeoLife preparation produced no valid trajectories: "
                f"{inspection.rejected_by_reason}"
            )
        ids = [record.trajectory_id for record in records]
        splits = make_split_bundle(
            ids,
            seed=int(settings.get("seed", 0)),
            records=records,
            include_user_held_out=True,
        )
        raw_checksums = {
            path.relative_to(root).as_posix(): sha256_file(path) for path in self._files(root)
        }
        output = write_canonical_dataset(
            output_path,
            records,
            dataset="geolife",
            version=str(settings.get("version", "v1")),
            projected_crs=str(settings.get("projected_crs", "EPSG:32650")),
            source_name="GeoLife GPS Trajectories",
            source_url=settings.get("source_url"),
            source_license=settings.get("source_license", "See the official GeoLife terms"),
            redistribution_policy="raw input is user-supplied and not redistributed",
            raw_checksums=raw_checksums,
            preprocessing_config_hash=settings.get("preprocessing_config_hash"),
            splits=splits,
            min_points=int(settings.get("min_points", 2)),
        )
        self.last_inspection = inspection
        return PreparationResult(output, inspection)

    def describe_license(self) -> Mapping[str, Any]:
        return {
            "dataset": "geolife",
            "status": "user-supplied raw input required",
            "redistribution": "raw data is not included or downloaded by this loader",
            "required_action": "check the official source terms before acquisition",
        }
