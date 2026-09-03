from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest

from trajsimbench.measures._geometry import (
    normalization_denominator,
    path_length,
    point_distances,
    timestamps,
    validate_pair,
)
from trajsimbench.measures._geometry import (
    projected_points as geometry_projected_points,
)
from trajsimbench.measures._geometry import (
    result as geometry_result,
)
from trajsimbench.measures.base import (
    DistanceResult,
    MeasureCapabilities,
    MeasureCapabilityError,
    TrajectoryMeasure,
    TrajectoryView,
    as_trajectory_view,
    projected_points,
)
from trajsimbench.measures.config import (
    CONFIG_MODELS,
    ConfigValidationError,
    DTWConfig,
    EDRConfig,
    ERPConfig,
    EuclideanConfig,
    LCSSConfig,
)
from trajsimbench.orchestration.cache import (
    cache_valid,
    file_fingerprint,
    fingerprint,
    load_stage_cache,
    save_stage_cache,
)
from trajsimbench.orchestration.resume import invalidate_from, resume_stage
from trajsimbench.perturbations import PERTURBATION_REGISTRY, get_perturbation
from trajsimbench.perturbations.base import (
    Perturbation,
    PerturbationError,
    geographic_columns,
    polyline_length,
    spatial_columns,
    timestamp_column,
    trajectory_input,
    validate_trajectory_points,
)
from trajsimbench.perturbations.registry import PerturbationRegistry
from trajsimbench.utils.logging import configure_logging
from trajsimbench.utils.paths import cache_dir, project_root, resolve_under
from trajsimbench.utils.seeding import seed_everything, seeded
from trajsimbench.utils.timing import Timer, timed_call


def canonical_points(count: int = 6) -> np.ndarray:
    values = np.arange(count, dtype=np.float64)
    xy = np.column_stack((values * 10.0, values * values))
    points = np.zeros((count, 5), dtype=np.float64)
    points[:, 0] = -73.0 + xy[:, 0] / 111_000.0
    points[:, 1] = 45.0 + xy[:, 1] / 111_000.0
    points[:, 2:4] = xy
    points[:, 4] = values * 10.0
    return points


def test_method_configs_validate_aliases_defaults_and_dependent_fields() -> None:
    euclidean = EuclideanConfig(sampling_count=np.int64(7))
    assert euclidean.n_samples == 7
    assert euclidean.sampling_count == euclidean.resample_count == 7
    assert EuclideanConfig.model_validate(euclidean) == euclidean
    assert EuclideanConfig.model_validate({"num_samples": 8}).model_dump()["n_samples"] == 8
    assert EuclideanConfig.fields() == ("n_samples",)
    assert EuclideanConfig(n_samples=3).model_copy(update={"n_samples": 4}).n_samples == 4
    assert "n_samples=3" in repr(EuclideanConfig(n_samples=3))
    assert EuclideanConfig(n_samples=3).dict() == {"n_samples": 3}
    assert EuclideanConfig(n_samples=3) == EuclideanConfig(n_samples=3)
    assert set(CONFIG_MODELS) == {
        "euclidean",
        "dtw",
        "hausdorff",
        "discrete_frechet",
        "lcss",
        "edr",
        "erp",
    }

    dtw = DTWConfig(global_normalization="path_length", sakoe_chiba_window=np.int64(2))
    assert dtw.normalization == dtw.global_normalization == "path_length"
    assert dtw.window == dtw.window_size == 2
    assert DTWConfig(window=None).window is None
    lcss = LCSSConfig(epsilon=np.float64(0.5), delta_mode="time", delta=1.5, time_delta=2)
    assert lcss.delta == 1.5 and lcss.time_delta_s == 2.0
    assert LCSSConfig(delta=2, delta_mode="index").delta == 2
    assert EDRConfig(epsilon=np.int64(2)).epsilon == 2.0
    erp = ERPConfig(gap=np.array([1, 2]), normalize=True)
    assert erp.gap_point == (1.0, 2.0) and erp.normalize
    assert ERPConfig(normalize=False).normalization == "none"
    for constructor, values in (
        (EuclideanConfig, {"n_samples": True}),
        (DTWConfig, {"normalization": "bad"}),
        (DTWConfig, {"window": -1}),
        (LCSSConfig, {"epsilon": float("nan")}),
        (LCSSConfig, {"delta_mode": "time", "delta": True}),
        (LCSSConfig, {"time_delta": -1}),
        (EDRConfig, {"epsilon": -1}),
        (ERPConfig, {"gap_point": [1]}),
        (ERPConfig, {"gap_point": [float("inf"), 0]}),
        (ERPConfig, {"normalize": 1}),
        (ERPConfig, {"normalize": True, "normalization": "none"}),
    ):
        with pytest.raises(ConfigValidationError):
            constructor(**values)
    with pytest.raises(ConfigValidationError, match="duplicate field"):
        EuclideanConfig(n_samples=3, sampling_count=4)
    with pytest.raises(ConfigValidationError, match="unknown field"):
        EuclideanConfig(unknown=1)
    with pytest.raises(TypeError, match="mapping or config"):
        EuclideanConfig.model_validate(3)


def test_geometry_helpers_cover_canonical_compact_and_metadata_paths() -> None:
    canonical = TrajectoryView("canonical", canonical_points())
    compact = TrajectoryView("compact", [[0.0, 0.0], [3.0, 4.0]])
    override = TrajectoryView(
        "override", canonical_points(), {"projected_points": [[2, 3, 99], [5, 7, 99]]}
    )
    np.testing.assert_array_equal(geometry_projected_points(canonical), canonical.points[:, 2:4])
    np.testing.assert_array_equal(geometry_projected_points(compact), [[0, 0], [3, 4]])
    np.testing.assert_array_equal(geometry_projected_points(override), [[2, 3], [5, 7]])
    np.testing.assert_array_equal(validate_pair(canonical, compact)[1], [[0, 0], [3, 4]])
    np.testing.assert_allclose(point_distances(np.array([[0, 0]]), np.array([[3, 4]])), [[5]])
    assert path_length(np.array([[0, 0], [3, 4]], dtype=float)) == 5.0
    assert path_length(np.array([[0, 0]], dtype=float)) == 0.0
    assert normalization_denominator("none", np.ones((1, 2)), np.ones((3, 2)), 0) == 1.0
    assert normalization_denominator("max_input_length", np.ones((1, 2)), np.ones((3, 2)), 0) == 3.0
    assert (
        normalization_denominator(
            "path_length", np.array([[0, 0], [3, 4.0]]), np.array([[0, 0]]), 0
        )
        == 5.0
    )
    assert geometry_result(2, details={"method": "x"}).details["method"] == "x"
    with pytest.raises(ValueError, match="unknown normalization"):
        normalization_denominator("bad", np.ones((1, 2)), np.ones((1, 2)), 0)
    with pytest.raises(ValueError, match="invalid distance"):
        geometry_result(-1)
    with pytest.raises(ValueError, match="invalid distance"):
        geometry_result(1, distance=float("nan"))
    for bad in (TrajectoryView("empty", np.empty((0, 2))), TrajectoryView("bad", [[1, np.nan]])):
        with pytest.raises(ValueError):
            geometry_projected_points(bad)
    with pytest.raises(ValueError, match="projected_points"):
        geometry_projected_points(
            TrajectoryView("bad", canonical_points(), {"projected_points": [1, 2]})
        )


def test_geometry_timestamps_and_measure_base_contracts() -> None:
    points = canonical_points()
    view = TrajectoryView("v", points)
    np.testing.assert_array_equal(timestamps(view), points[:, 4])
    np.testing.assert_array_equal(timestamps(TrajectoryView("v", points[:, :3])), points[:, 2])
    np.testing.assert_array_equal(
        timestamps(TrajectoryView("v", points[:, :2], {"timestamps_s": [1, 2, 3, 4, 5, 6]})),
        [1, 2, 3, 4, 5, 6],
    )
    assert (
        timestamps(TrajectoryView("v", points[:, :2], {"timestamp_s": [1, np.nan, 3, 4, 5, 6]}))
        is None
    )
    assert timestamps(TrajectoryView("v", points[:, :2])) is None
    with pytest.raises(ValueError, match="align"):
        timestamps(TrajectoryView("v", points[:, :2], {"timestamps": [1]}))
    assert (
        as_trajectory_view({"id": "mapping", "points": [[0, 0], [1, 1]]}).trajectory_id == "mapping"
    )
    assert (
        as_trajectory_view(
            type("Obj", (), {"id": "object", "points": [[0, 0], [1, 1]]})()
        ).trajectory_id
        == "object"
    )
    assert as_trajectory_view([[0, 0], [1, 1]]).trajectory_id == "trajectory"
    assert projected_points(np.empty((0, 4)), allow_empty=True).shape == (0, 2)
    with pytest.raises(ValueError, match="numeric"):
        geometry_projected_points(TrajectoryView("bad", np.array([["x", "y"]], dtype=object)))
    with pytest.raises(ValueError, match="two-dimensional"):
        TrajectoryView("bad", [1, 2])
    with pytest.raises(TypeError, match="mapping"):
        TrajectoryView("bad", [[1, 2]], metadata=[])

    class BatchMeasure(TrajectoryMeasure):
        name = "test_measure"
        capabilities = MeasureCapabilities(
            supports_batch=True, supports_encoding=True, supports_index=True
        )

        def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
            left, right = validate_pair(a, b)
            return DistanceResult(float(np.linalg.norm(left[0] - right[0])), 2.0)

    measure = BatchMeasure()
    assert measure.fit([], []) is measure
    assert measure.metadata["name"] == "test_measure"
    assert measure.pairwise([[0, 0]], np.array([[[0, 0]], [[1, 0]]], dtype=float)).shape == (2,)
    assert measure.pairwise([[0, 0]], np.array([[0, 0]], dtype=float)).shape == (1,)
    assert measure.distance({"points": [[0, 0], [1, 0]]}, [[0, 0], [1, 0]]).runtime_ns is not None
    with pytest.raises(MeasureCapabilityError, match="encoding"):
        measure.encode([])
    with pytest.raises(MeasureCapabilityError, match="indexing"):
        measure.build_index([])
    with pytest.raises(MeasureCapabilityError, match="advertises indexing"):
        measure.top_k([], 1)
    with pytest.raises(ValueError, match="non-negative"):
        DistanceResult(-1, 0)
    with pytest.raises(ValueError, match="finite"):
        DistanceResult(1, np.inf)
    with pytest.raises(TypeError, match="runtime_ns"):
        DistanceResult(1, 0, runtime_ns=True)
    with pytest.raises(ValueError, match="non-negative"):
        DistanceResult(1, 0, runtime_ns=-1)
    with pytest.raises(TypeError, match="mapping"):
        DistanceResult(1, 0, details=[])


def test_perturbation_registry_resolution_specs_and_regeneration() -> None:
    registry = PerturbationRegistry()
    registry.register("noise", PERTURBATION_REGISTRY.get("gps_noise").__class__, aliases=("n",))
    assert registry.names() == ("noise",)
    assert registry.resolve_name(" N ") == "noise"
    assert isinstance(registry.create("noise"), Perturbation)
    assert isinstance(registry.create({"type": "noise", "config": {}}), Perturbation)
    source = canonical_points()
    applied = registry.apply("noise", source, severity=1.0, seed=4)
    assert applied.generated
    assert get_perturbation("gps_drift", rho=0.5).name == "gps_drift"
    with pytest.raises(PerturbationError, match="duplicate"):
        registry.register("noise", PERTURBATION_REGISTRY.get("gps_noise").__class__)
    with pytest.raises(PerturbationError, match="alias"):
        registry.register("other", PERTURBATION_REGISTRY.get("gps_noise").__class__, aliases=("n",))
    with pytest.raises(KeyError, match="unknown perturbation"):
        registry.get("missing")
    with pytest.raises(PerturbationError, match="requires"):
        registry.create({"config": {}})
    with pytest.raises(PerturbationError, match="name, mapping"):
        registry.create(3)
    with pytest.raises(PerturbationError, match="extra config"):
        get_perturbation(applied, sigma=1.0)

    cases = [
        ("gps_drift", 1.0),
        ("random_point_loss", 0.7),
        ("contiguous_outage", 0.2),
        ("sampling_reduction", {"mode": "ratio", "value": 0.7}),
        ("spatial_quantization", 2.0),
        ("temporal_jitter", {"scale_s": 0.1, "distribution": "uniform"}),
        ("truncation", {"fraction": 0.1, "side": "both"}),
        ("reversal", {"timestamp_policy": "omit"}),
        ("spatial_translation", {"magnitude_m": 5.0, "bearing_rad": 0.2}),
        ("free_space_detour", 0.1),
    ]
    for name, severity in cases:
        result = PERTURBATION_REGISTRY.apply(name, source, severity=severity, seed=7)
        if result.generated:
            regenerated = PERTURBATION_REGISTRY.regenerate(source, result.provenance)
            np.testing.assert_array_equal(result.points, regenerated.points)
            assert (
                PERTURBATION_REGISTRY.regenerate(source, result.provenance.to_dict()).variant_id
                == result.variant_id
            )


def test_temporal_and_sampling_perturbations_cover_real_modes_and_rejections() -> None:
    source = canonical_points(8)
    for distribution in ("normal", "uniform", "laplace"):
        result = PERTURBATION_REGISTRY.apply(
            "temporal_jitter",
            source,
            severity={"scale": 0.0, "distribution": distribution, "repair": "nondecreasing"},
            seed=1,
        )
        assert result.generated
    repaired = None
    for seed in range(100):
        candidate = PERTURBATION_REGISTRY.apply("temporal_jitter", source, severity=5.0, seed=seed)
        if candidate.generated and candidate.provenance.parameters.get("repair_count", 0) > 0:
            repaired = candidate
            break
    assert repaired is not None
    assert PERTURBATION_REGISTRY.apply(
        "temporal_jitter", source[:, :4], severity=1.0, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply(
        "temporal_jitter", source, severity={"repair": "sort", "scale": 1}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply(
        "temporal_jitter", source, severity={"repair": "bad", "scale": 1}, seed=1
    ).reason

    speed = PERTURBATION_REGISTRY.apply(
        "speed_distortion", source, severity={"piecewise": [1, 2, 3, 4, 5, 6, 7]}, seed=1
    )
    assert speed.generated and speed.points[-1, 4] > source[-1, 4]
    assert PERTURBATION_REGISTRY.apply("speed_distortion", source[:, :4], severity=2, seed=1).reason
    assert PERTURBATION_REGISTRY.apply(
        "speed_distortion", source, severity={"piecewise": [1]}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply("speed_distortion", source, severity=0, seed=1).reason

    for policy in ("rebase", "reverse_durations", "omit"):
        result = PERTURBATION_REGISTRY.apply(
            "reversal", source, severity={"timestamp_policy": policy}, seed=1
        )
        assert result.generated
        if policy == "omit":
            assert np.isnan(result.points[:, 4]).all()
    assert PERTURBATION_REGISTRY.apply(
        "reversal", source, severity={"timestamp_policy": "bad"}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply("reversal", source[:, :4], severity=1, seed=1).generated

    ratio = PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "ratio", "value": 0.5}, seed=1
    )
    time = PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "seconds", "interval": 15}, seed=1
    )
    spatial = PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "meters", "value": 15}, seed=1
    )
    no_endpoints = PERTURBATION_REGISTRY.apply("sampling_reduction", source, severity=0.5, seed=1)
    assert all(item.generated for item in (ratio, time, spatial, no_endpoints))
    assert PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source[:, :4], severity={"mode": "time", "value": 1}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "bad", "value": 1}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "ratio"}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply(
        "sampling_reduction", source, severity={"mode": "ratio", "value": 2}, seed=1
    ).reason
    assert PERTURBATION_REGISTRY.apply("sampling_reduction", source, severity=0, seed=1).reason
    assert PERTURBATION_REGISTRY.apply("sampling_reduction", source, severity=0.1, seed=1).reason

    loss = PERTURBATION_REGISTRY.apply("random_point_loss", source, severity={"ratio": 0.5}, seed=1)
    loss_any = PERTURBATION_REGISTRY.get("random_point_loss", preserve_endpoints=False).apply(
        source, severity=0.5, seed=1
    )
    outage = PERTURBATION_REGISTRY.apply(
        "contiguous_outage", source, severity={"value": 0.2}, seed=1
    )
    assert loss.generated and loss_any.generated and outage.generated
    assert PERTURBATION_REGISTRY.apply("random_point_loss", source, severity=0.1, seed=1).reason
    assert PERTURBATION_REGISTRY.apply("contiguous_outage", source, severity=0.99, seed=1).reason
    for side in ("start", "end", "both"):
        assert (
            PERTURBATION_REGISTRY.get("truncation", side=side)
            .apply(source, severity=0.2, seed=1)
            .generated
        )
    assert PERTURBATION_REGISTRY.apply(
        "truncation", source, severity={"fraction": 0.2, "side": "bad"}, seed=1
    ).reason


def test_perturbation_base_helpers_validate_metadata_and_copy_updates() -> None:
    points = canonical_points()
    view = trajectory_input(
        type("Obj", (), {"trajectory_id": "x", "points": points, "metadata": {}})()
    )
    assert view.points.flags.writeable is False
    assert spatial_columns(points) == (2, 3)
    assert geographic_columns(points) == (0, 1)
    assert timestamp_column(points) == 4
    assert timestamp_column(points[:, :2]) is None
    assert timestamp_column(points[:, :3], {"timestamp_column": 2}) == 2
    assert polyline_length(points) > 0
    validate_trajectory_points(points)
    with pytest.raises(PerturbationError, match="2-D"):
        trajectory_input(np.array([1, 2]))
    with pytest.raises(PerturbationError, match="spatial columns"):
        spatial_columns(np.ones((2, 2)), {"projected_columns": (2, 3)})
    with pytest.raises(PerturbationError, match="at least"):
        validate_trajectory_points(points[:1], min_points=2)
    bad = points.copy()
    bad[0, 2] = np.nan
    with pytest.raises(PerturbationError, match="projected"):
        validate_trajectory_points(bad)
    bad_geo = points.copy()
    bad_geo[0, 0] = 200
    with pytest.raises(PerturbationError, match="longitude"):
        validate_trajectory_points(bad_geo)
    bad_time = points.copy()
    bad_time[2, 4] = 1
    with pytest.raises(PerturbationError, match="monotonic"):
        validate_trajectory_points(bad_time)


def test_orchestration_cache_resume_and_utils_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    assert file_fingerprint([second, first]) == file_fingerprint([first, second])
    missing_fingerprint = file_fingerprint([tmp_path / "missing"])
    assert missing_fingerprint != file_fingerprint([first])
    cache_path = tmp_path / "cache.json"
    records = {"stage": {"status": "complete", "input_fingerprint": "fp", "outputs": ["a.txt"]}}
    save_stage_cache(cache_path, records)
    assert load_stage_cache(cache_path) == records
    assert cache_valid(records["stage"], input_fingerprint="fp", root=tmp_path)
    assert not cache_valid(records["stage"], input_fingerprint="wrong", root=tmp_path)
    assert not cache_valid({"status": "pending"}, input_fingerprint="fp", root=tmp_path)
    assert not cache_valid(
        {"status": "complete", "input_fingerprint": "fp", "outputs": ["missing"]},
        input_fingerprint="fp",
        root=tmp_path,
    )
    cache_path.write_text("broken", encoding="utf-8")
    assert load_stage_cache(cache_path) == {}
    assert load_stage_cache(tmp_path / "no-cache.json") == {}
    state = {
        "load_data": {"status": "complete"},
        "metrics": {"status": "complete"},
        "other": {"status": "complete"},
    }
    assert invalidate_from(state, "load_data")["metrics"]["status"] == "pending"
    assert state["other"]["status"] == "complete"
    assert resume_stage(records, "stage", input_fingerprint="fp", root=tmp_path)

    assert project_root(tmp_path) == tmp_path
    assert project_root(Path.cwd()).name == "TrajSimBench"
    monkeypatch.setenv("TRAJSIMBENCH_CACHE", str(tmp_path / "cache"))
    assert cache_dir("test-app") == tmp_path / "cache"
    assert resolve_under(tmp_path, "nested/file.txt") == tmp_path / "nested/file.txt"
    with pytest.raises(ValueError, match="escapes"):
        resolve_under(tmp_path, "../outside.txt")

    seed_everything(11)
    expected = (random.random(), np.random.random())
    seed_everything(11)
    assert (random.random(), np.random.random()) == expected
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    with seeded(3) as generator:
        inside = (generator.random(), random.random(), np.random.random())
        assert inside[0] != expected[0]
    assert random.getstate() == random_before
    restored = np.random.get_state()
    assert restored[0] == numpy_before[0] and np.array_equal(restored[1], numpy_before[1])

    with Timer() as timer:
        value = sum(range(10))
    assert value == 45 and timer.elapsed_ns >= 0
    assert timed_call(lambda x: x + 1, 2)[0] == 3
    logger = configure_logging(logging.DEBUG)
    assert logger.name == "trajsimbench"
