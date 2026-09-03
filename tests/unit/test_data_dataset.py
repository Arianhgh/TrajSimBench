from pathlib import Path

import numpy as np
import pytest

from trajsimbench.data.dataset import TrajectoryDataset, TrajectoryInput, write_canonical_dataset
from trajsimbench.data.projection import inverse_project_coordinates, project_coordinates
from trajsimbench.data.validation import DatasetValidationError, validate_dataset


def _records() -> list[TrajectoryInput]:
    return [
        TrajectoryInput(
            "fixture:a",
            np.array([[13.4, 52.5, 0.0], [13.401, 52.5, 1.0], [13.402, 52.501, 2.0]]),
            user_id="u1",
        ),
        TrajectoryInput(
            "fixture:b",
            np.array([[13.5, 52.5], [13.501, 52.5]]),
            user_id="u2",
        ),
    ]


def test_round_trip_mmap_views_and_splits(tmp_path: Path) -> None:
    output = write_canonical_dataset(
        tmp_path / "dataset",
        _records(),
        dataset="fixture",
        version="v1",
        projected_crs="EPSG:32633",
        splits={"standard": {"train": ["fixture:a"], "test": ["fixture:b"]}},
        feature_arrays={"speed_mps": np.arange(5, dtype=np.float64)},
    )
    report = validate_dataset(output)
    assert report.ok, report.errors
    dataset = TrajectoryDataset.open(output)
    assert len(dataset) == 2
    assert dataset.ids("train").tolist() == ["fixture:a"]
    assert dataset.by_id("fixture:a").points.shape == (3, 5)
    assert dataset.feature("speed_mps").tolist() == list(range(5))
    assert not dataset[0].points.flags.writeable
    with pytest.raises(ValueError):
        dataset[0].points[0, 0] = 0
    assert dataset[-1].trajectory_id == "fixture:b"


def test_corrupt_content_fails_checksum_validation(tmp_path: Path) -> None:
    output = write_canonical_dataset(
        tmp_path / "dataset",
        _records(),
        dataset="fixture",
        projected_crs="EPSG:32633",
    )
    with (output / "points.npy").open("ab") as handle:
        handle.write(b"corruption")
    report = validate_dataset(output)
    assert not report.ok
    assert any("checksum mismatch" in error for error in report.errors)
    with pytest.raises(DatasetValidationError):
        report.raise_if_invalid()


def test_projection_round_trip_is_metric_and_xy_ordered() -> None:
    lon = np.array([13.4, 13.41])
    lat = np.array([52.5, 52.51])
    x, y = project_coordinates(lon, lat, "EPSG:32633")
    lon_back, lat_back = inverse_project_coordinates(x, y, "EPSG:32633")
    assert np.allclose(lon, lon_back, atol=1e-9)
    assert np.allclose(lat, lat_back, atol=1e-9)
    assert np.linalg.norm([x[1] - x[0], y[1] - y[0]]) > 0
