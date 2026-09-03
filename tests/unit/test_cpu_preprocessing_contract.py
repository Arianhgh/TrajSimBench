from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trajsimbench.data.preprocessing import (
    clean_points,
    deduplicate_consecutive,
    resample_polyline,
    trajectory_statistics,
)
from trajsimbench.data.projection import (
    choose_local_utm,
    inverse_project_coordinates,
    project_coordinates,
    projection_round_trip_error,
)
from trajsimbench.data.splitting import (
    make_split_bundle,
    select_scale,
    stable_id_order,
    standard_split,
    temporal_split,
    user_held_out_split,
)


def test_cleaning_reports_nonfinite_and_consecutive_duplicates() -> None:
    points = np.array([[0.0, 1.0], [0.0, 1.0], [np.nan, 2.0], [3.0, 4.0]])
    cleaned, report = clean_points(points)
    np.testing.assert_array_equal(cleaned, [[0.0, 1.0], [3.0, 4.0]])
    assert report == {
        "input_points": 4,
        "dropped_nonfinite": 1,
        "deduplicated": 1,
        "output_points": 2,
    }
    np.testing.assert_array_equal(deduplicate_consecutive(np.array([[1.0, 2.0]])), [[1.0, 2.0]])
    with pytest.raises(ValueError, match="two-dimensional"):
        clean_points(np.array([1.0, 2.0]))


def test_clean_points_options_preserve_requested_rows() -> None:
    points = np.array([[0.0, 1.0], [0.0, 1.0], [np.nan, 2.0]])
    kept, report = clean_points(points, deduplicate=False, drop_nonfinite=False)
    assert kept.shape == points.shape
    assert np.all((kept == points) | (np.isnan(kept) & np.isnan(points)))
    assert report["dropped_nonfinite"] == 0
    assert report["deduplicated"] == 0


def test_resampling_interpolates_all_columns_and_rejects_bad_time_axes() -> None:
    points = np.array([[0.0, 10.0], [10.0, 30.0], [20.0, 50.0]])
    result = resample_polyline(points, [0.0, 2.0, 4.0], [-1.0, 1.0, 3.0, 5.0])
    np.testing.assert_allclose(result, [[0.0, 10.0], [5.0, 20.0], [15.0, 40.0], [20.0, 50.0]])
    with pytest.raises(ValueError, match="matching row counts"):
        resample_polyline(points, [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="non-empty"):
        resample_polyline(np.empty((0, 2)), [], [0.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        resample_polyline(points, [0.0, 2.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="unique"):
        resample_polyline(points, [0.0, 1.0, 1.0], [0.0])


def test_trajectory_statistics_handles_time_and_single_point_cases() -> None:
    points = np.array([[0.0, 0.0, 10.0], [3.0, 4.0, np.nan], [3.0, 8.0, 25.0]])
    stats = trajectory_statistics(points)
    assert stats == {
        "num_points": 3,
        "length_m": 9.0,
        "start_time_s": 10.0,
        "end_time_s": 25.0,
        "duration_s": 15.0,
    }
    assert trajectory_statistics(np.array([[1.0, 2.0]]))["length_m"] == 0.0
    assert trajectory_statistics(np.array([[1.0, 2.0, np.nan]]))["duration_s"] is None
    with pytest.raises(ValueError, match="at least two"):
        trajectory_statistics(np.ones((2, 1)))


def test_projection_chooses_northern_and_southern_utm_and_validates_extent() -> None:
    assert choose_local_utm(np.array([-73.0, -72.0]), np.array([45.0, 46.0])) == "EPSG:32618"
    assert choose_local_utm(np.array([151.0, 151.5]), np.array([-33.0, -34.0])) == "EPSG:32756"
    with pytest.raises(ValueError, match="at least one"):
        choose_local_utm(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="one UTM zone"):
        choose_local_utm(np.array([-80.0, -70.0]), np.array([40.0, 41.0]))


def test_projection_round_trip_and_shape_validation() -> None:
    lon = np.array([-73.1, -73.0])
    lat = np.array([45.4, 45.5])
    x, y = project_coordinates(lon, lat, "EPSG:32618")
    lon_back, lat_back = inverse_project_coordinates(x, y, "EPSG:32618")
    np.testing.assert_allclose(lon_back, lon, atol=1e-10)
    np.testing.assert_allclose(lat_back, lat, atol=1e-10)
    assert projection_round_trip_error(lon, lat, "EPSG:32618") < 1e-8
    with pytest.raises(ValueError, match="matching shapes"):
        project_coordinates([1.0], [2.0, 3.0], "EPSG:32618")
    with pytest.raises(ValueError, match="matching shapes"):
        inverse_project_coordinates([1.0], [2.0, 3.0], "EPSG:32618")


def test_splits_are_deterministic_and_support_records_and_options() -> None:
    ids = ["z", "a", "m", "b", "q"]
    assert stable_id_order(ids, seed=7) == stable_id_order(ids, seed=7)
    split = standard_split(ids, seed=7, ratios=(0.4, 0.2, 0.4))
    assert sorted(sum(split.values(), [])) == sorted(ids)
    assert set(split["train"]).isdisjoint(split["val"])
    records = [
        {"trajectory_id": "a", "user_id": "u1", "start_time_s": 3},
        SimpleNamespace(trajectory_id="b", user_id="u2", start_time_s=1, metadata={}),
        {"trajectory_id": "c", "user_id": "u1", "start_time_s": 2},
    ]
    held_out = user_held_out_split(records, ratios=(0.5, 0.0, 0.5), seed=2)
    user_by_id = {"a": "u1", "b": "u2", "c": "u1"}
    users = [{user_by_id[item] for item in values} for values in held_out.values()]
    assert not (users[0] & users[1] or users[0] & users[2] or users[1] & users[2])
    temporal = temporal_split(records, ratios=(0.5, 0.0, 0.5))
    assert temporal["train"] == ["b", "c"]
    with pytest.raises(ValueError, match="user_id"):
        user_held_out_split([{"trajectory_id": "x", "user_id": None}])
    with pytest.raises(ValueError, match="start_time_s"):
        temporal_split([{"trajectory_id": "x", "start_time_s": np.nan}])


def test_split_bundle_and_scale_selection_make_reductions_explicit() -> None:
    records = [
        {"trajectory_id": "a", "user_id": "u1", "start_time_s": 1.0},
        {"trajectory_id": "b", "user_id": "u2", "start_time_s": 2.0},
    ]
    bundle = make_split_bundle(
        ["a", "b"], records=records, include_temporal=True, include_user_held_out=True
    )
    assert set(bundle) == {"standard", "temporal", "user_held_out"}
    with pytest.raises(ValueError, match="records are required"):
        make_split_bundle(["a"], include_temporal=True)
    selection = select_scale(["a", "b", "c"], "tiny", seed=1, database_count=2, query_count=1)
    assert len(selection.database_ids) == 2
    assert len(selection.query_ids) == 1
    with pytest.raises(ValueError, match="needs"):
        select_scale(["a", "b"], "tiny", database_count=2, query_count=1)
    reduced = select_scale(["a", "b"], "tiny", database_count=2, query_count=2, allow_reduced=True)
    assert len(reduced.database_ids) == 2 and len(reduced.query_ids) == 0
    with pytest.raises(ValueError, match="unknown scale"):
        select_scale(["a"], "unknown")
