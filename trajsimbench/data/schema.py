"""Names and lightweight records for the canonical trajectory format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

POINT_COLUMNS = ("lon_deg", "lat_deg", "x_m", "y_m", "timestamp_s")
SCHEMA_VERSION = "1.0"
REQUIRED_METADATA_COLUMNS = (
    "trajectory_idx",
    "trajectory_id",
    "dataset",
    "source_id",
    "user_id",
    "start_time_s",
    "end_time_s",
    "mobility_mode",
    "length_m",
    "duration_s",
    "num_points",
    "split",
    "crs_projected",
    "quality_flags",
)


@dataclass(frozen=True, slots=True)
class TrajectoryInput:
    """WGS84 trajectory input accepted by all preparation loaders.

    ``points`` is ``[n, 2]`` for longitude/latitude or ``[n, 3]`` for
    longitude/latitude/Unix-seconds. A five-column canonical array is also
    accepted when importing an already projected fixture; its projected
    columns are recomputed by the writer to enforce the configured CRS.
    """

    trajectory_id: str
    points: np.ndarray
    source_id: str | None = None
    user_id: str | None = None
    mobility_mode: str | None = None
    quality_flags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] not in (2, 3, 5):
            raise ValueError("TrajectoryInput.points must have shape [n,2], [n,3], or [n,5]")
        if not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DataSetInfo:
    """Metadata written to ``dataset.json``."""

    dataset: str
    version: str
    projected_crs: str
    source_name: str | None = None
    source_url: str | None = None
    source_license: str | None = None
    redistribution_policy: str | None = None
