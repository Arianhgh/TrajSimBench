from types import SimpleNamespace

import numpy as np
import pytest

from trajsimbench.orchestration.cache import (
    cache_valid,
    file_fingerprint,
    fingerprint,
    load_stage_cache,
    save_stage_cache,
)
from trajsimbench.orchestration.resume import invalidate_from, resume_stage
from trajsimbench.perturbations import PERTURBATION_REGISTRY
from trajsimbench.perturbations.base import PerturbationError
from trajsimbench.perturbations.registry import PerturbationRegistry, get_perturbation
from trajsimbench.perturbations.sampling import (
    ContiguousOutagePerturbation,
    RandomPointLossPerturbation,
    SamplingFrequencyReductionPerturbation,
    TruncationPerturbation,
)
from trajsimbench.perturbations.temporal import (
    ReversalPerturbation,
    SpeedDistortionPerturbation,
    TemporalJitterPerturbation,
)
from trajsimbench.utils.logging import configure_logging
from trajsimbench.utils.paths import cache_dir, project_root, resolve_under
from trajsimbench.utils.seeding import seeded
from trajsimbench.utils.timing import Timer, timed_call


def _source(n: int = 8) -> SimpleNamespace:
    points = np.zeros((n, 5), dtype=np.float64)
    points[:, 0] = -73.0 + np.arange(n) / 111_000.0
    points[:, 1] = 45.0
    points[:, 2] = np.arange(n, dtype=np.float64) * 10.0
    points[:, 3] = np.arange(n, dtype=np.float64) * 2.0
    points[:, 4] = np.arange(n, dtype=np.float64) * 5.0
    return SimpleNamespace(trajectory_id="unit-source", points=points, metadata={})


def test_temporal_jitter_distributions_repairs_and_rejections():
    source = _source(4)
    for distribution in ("normal", "uniform", "laplace"):
        result = TemporalJitterPerturbation(distribution=distribution).apply(
            source,
            severity={"scale_s": 0.0, "distribution": distribution, "repair": "nondecreasing"},
            seed=1,
        )
        assert result.generated
        np.testing.assert_array_equal(result.points[:, 4], source.points[:, 4])
        assert result.provenance.parameters["distribution"] == distribution

    missing_time = TemporalJitterPerturbation().apply(np.ones((3, 2)), severity=1.0, seed=1)
    assert missing_time.status == "not_generated"
    assert "timestamp column" in missing_time.reason
    bad_time = _source(3)
    bad_time.points[1, 4] = np.nan
    rejected = TemporalJitterPerturbation().apply(bad_time, severity=1.0, seed=1)
    assert rejected.status == "not_generated"
    assert "finite monotonic" in rejected.reason
    for severity in (
        -1.0,
        {"scale": 1.0, "distribution": "other"},
        {"scale": 1.0, "repair": "sort"},
    ):
        assert (
            TemporalJitterPerturbation().apply(source, severity=severity, seed=1).status
            == "not_generated"
        )


def test_speed_distortion_scalar_piecewise_and_errors():
    source = _source(4)
    scalar = SpeedDistortionPerturbation().apply(source, severity=2.0, seed=2)
    np.testing.assert_allclose(scalar.points[:, 4], [0.0, 10.0, 20.0, 30.0])
    piecewise = SpeedDistortionPerturbation().apply(
        source, severity={"piecewise": [1.0, 2.0, 3.0]}, seed=2
    )
    np.testing.assert_allclose(piecewise.points[:, 4], [0.0, 5.0, 15.0, 30.0])
    assert (
        SpeedDistortionPerturbation().apply(np.ones((2, 2)), severity=2.0, seed=2).status
        == "not_generated"
    )
    for severity in (0.0, {"piecewise": [1.0]}, {"piecewise": [1.0, 0.0, 1.0]}):
        assert (
            SpeedDistortionPerturbation().apply(source, severity=severity, seed=2).status
            == "not_generated"
        )
    descending = _source(3)
    descending.points[2, 4] = 1.0
    assert (
        "monotonic" in SpeedDistortionPerturbation().apply(descending, severity=2.0, seed=2).reason
    )


def test_reversal_policies_preserve_or_omit_timestamps():
    source = _source(4)
    rebased = ReversalPerturbation().apply(source, severity=0.0, seed=3)
    np.testing.assert_array_equal(rebased.points[:, :4], source.points[::-1, :4])
    np.testing.assert_allclose(rebased.points[:, 4], [0.0, 5.0, 10.0, 15.0])
    omitted = ReversalPerturbation().apply(source, severity={"timestamp_policy": "omit"}, seed=3)
    assert np.isnan(omitted.points[:, 4]).all()
    durations = ReversalPerturbation().apply(
        source, severity={"timestamp_policy": "reverse_durations"}, seed=3
    )
    np.testing.assert_allclose(np.diff(durations.points[:, 4]), np.diff(source.points[:, 4]))
    for policy in ("invalid",):
        assert (
            ReversalPerturbation()
            .apply(source, severity={"timestamp_policy": policy}, seed=3)
            .status
            == "not_generated"
        )
    bad = _source(3)
    bad.points[1, 4] = np.nan
    assert "monotonic" in ReversalPerturbation().apply(bad, severity=0.0, seed=3).reason


def test_sampling_perturbations_cover_modes_and_validation():
    source = _source(8)
    kept = RandomPointLossPerturbation().apply(source, severity={"ratio": 0.5}, seed=4)
    assert kept.generated and len(kept.points) == 4
    assert kept.points[0, 4] == source.points[0, 4]
    no_endpoints = RandomPointLossPerturbation(preserve_endpoints=False).apply(
        source, severity=0.5, seed=4
    )
    assert no_endpoints.generated and len(no_endpoints.points) == 4
    assert (
        RandomPointLossPerturbation().apply(source, severity=0.1, seed=4).status == "not_generated"
    )
    assert (
        RandomPointLossPerturbation().apply(source, severity=0.0, seed=4).status == "not_generated"
    )

    outage = ContiguousOutagePerturbation().apply(source, severity={"fraction": 0.25}, seed=5)
    assert outage.generated and len(outage.points) == 6
    assert outage.points[0, 4] == source.points[0, 4]
    no_endpoint_outage = ContiguousOutagePerturbation(preserve_endpoints=False).apply(
        source, severity=0.25, seed=5
    )
    assert no_endpoint_outage.generated
    assert (
        ContiguousOutagePerturbation().apply(source, severity=0.99, seed=5).status
        == "not_generated"
    )

    ratio = SamplingFrequencyReductionPerturbation().apply(source, severity=0.5, seed=6)
    assert ratio.generated and len(ratio.points) == 4
    all_points = SamplingFrequencyReductionPerturbation().apply(source, severity=1.0, seed=6)
    assert len(all_points.points) == len(source.points)
    no_endpoints_ratio = SamplingFrequencyReductionPerturbation(preserve_endpoints=False).apply(
        source, severity=0.5, seed=6
    )
    assert no_endpoints_ratio.generated
    temporal = SamplingFrequencyReductionPerturbation(mode="time").apply(
        source, severity={"interval": 10.0}, seed=6
    )
    assert temporal.generated and temporal.points[0, 4] == 0.0
    spatial = SamplingFrequencyReductionPerturbation(mode="spatial").apply(
        source, severity=15.0, seed=6
    )
    assert spatial.generated
    assert (
        SamplingFrequencyReductionPerturbation(mode="time")
        .apply(np.ones((4, 2)), severity=2.0, seed=6)
        .status
        == "not_generated"
    )
    assert (
        SamplingFrequencyReductionPerturbation().apply(source, severity=0.1, seed=6).status
        == "not_generated"
    )
    for severity in (None, 0.0, 2.0, {"mode": "unknown", "value": 1.0}):
        assert (
            SamplingFrequencyReductionPerturbation().apply(source, severity=severity, seed=6).status
            == "not_generated"
        )

    for side, expected in (
        ("start", source.points[2:]),
        ("end", source.points[:-2]),
        ("both", source.points[1:-1]),
    ):
        truncated = TruncationPerturbation(side=side).apply(source, severity=0.25, seed=7)
        assert truncated.generated
        np.testing.assert_array_equal(truncated.points, expected)
    assert TruncationPerturbation().apply(source, severity=0.9, seed=7).status == "not_generated"
    assert (
        TruncationPerturbation().apply(source, severity={"fraction": 0.0}, seed=7).status
        == "not_generated"
    )
    assert (
        TruncationPerturbation()
        .apply(source, severity={"fraction": 0.25, "side": "bad"}, seed=7)
        .status
        == "not_generated"
    )


def test_registry_specs_aliases_errors_and_regeneration():
    registry = PerturbationRegistry()
    registry.register("loss", RandomPointLossPerturbation, aliases=("drop",))
    assert registry.names() == ("loss",)
    assert registry.resolve_name(" DROP ") == "loss"
    assert isinstance(
        registry.create({"type": "loss", "preserve_endpoints": False}), RandomPointLossPerturbation
    )
    with pytest.raises(PerturbationError, match="duplicate"):
        registry.register("loss", RandomPointLossPerturbation)
    with pytest.raises(PerturbationError, match="duplicate"):
        registry.register("other", RandomPointLossPerturbation, aliases=("drop",))
    with pytest.raises(PerturbationError, match="invalid"):
        registry.register(" ", RandomPointLossPerturbation)
    with pytest.raises(KeyError, match="unknown perturbation"):
        registry.get("missing")
    with pytest.raises(PerturbationError, match="requires"):
        registry.create({"config": {}})
    with pytest.raises(PerturbationError, match="must be"):
        registry.create(12)
    registry.register("bad", lambda: object())
    with pytest.raises(PerturbationError, match="did not"):
        registry.get("bad")
    with pytest.raises(PerturbationError, match="extra config"):
        get_perturbation({"name": "loss"}, preserve_endpoints=True)

    source = _source()
    specs = {
        "gps_drift": 0.1,
        "random_point_loss": 0.75,
        "contiguous_outage": 0.1,
        "sampling_reduction": 0.75,
        "spatial_quantization": 10.0,
        "temporal_jitter": 0.0,
        "truncation": 0.1,
        "reversal": {"timestamp_policy": "omit"},
        "spatial_translation": {"magnitude_m": 1.0, "bearing_rad": 0.2},
        "free_space_detour": 0.1,
    }
    for name, severity in specs.items():
        first = PERTURBATION_REGISTRY.apply(name, source, severity=severity, seed=8)
        assert first.generated, (name, first.reason)
        regenerated = PERTURBATION_REGISTRY.regenerate(source, first.provenance.to_dict())
        np.testing.assert_array_equal(first.points, regenerated.points)


def test_cache_resume_and_utility_contracts(tmp_path, monkeypatch, caplog):
    payload = {"b": 2, "a": 1}
    assert fingerprint(payload) == fingerprint({"a": 1, "b": 2})
    existing = tmp_path / "existing.txt"
    existing.write_text("data", encoding="utf-8")
    assert file_fingerprint([tmp_path / "missing", existing]) == file_fingerprint(
        [existing, tmp_path / "missing"]
    )
    records = {
        "stage": {"status": "complete", "input_fingerprint": "abc", "outputs": ["existing.txt"]}
    }
    assert cache_valid(records["stage"], input_fingerprint="abc", root=tmp_path)
    assert not cache_valid(records["stage"], input_fingerprint="wrong", root=tmp_path)
    records["stage"]["status"] = "pending"
    assert not cache_valid(records["stage"], input_fingerprint="abc", root=tmp_path)
    records["stage"]["status"] = "complete"
    records["stage"]["outputs"] = ["missing"]
    assert not resume_stage(records, "stage", input_fingerprint="abc", root=tmp_path)
    cache_path = tmp_path / "cache.json"
    save_stage_cache(cache_path, records)
    assert load_stage_cache(cache_path) == records
    cache_path.write_text("{bad", encoding="utf-8")
    assert load_stage_cache(cache_path) == {}
    assert load_stage_cache(tmp_path / "none.json") == {}
    invalidate_from({"stage": {"status": "complete"}, "evaluate": {"status": "complete"}}, "stage")
    assert records["stage"]["status"] == "complete"

    assert project_root(tmp_path / "child") == tmp_path / "child"
    monkeypatch.setenv("TRAJSIMBENCH_CACHE", str(tmp_path / "cache-dir"))
    assert cache_dir() == tmp_path / "cache-dir"
    assert resolve_under(tmp_path, "nested/file.txt") == tmp_path / "nested/file.txt"
    with pytest.raises(ValueError, match="escapes"):
        resolve_under(tmp_path, "../outside")

    with seeded(123) as generator:
        inside = (np.random.random(), generator.integers(0, 100))
    with seeded(123) as again:
        assert inside == (np.random.random(), again.integers(0, 100))
    logger = configure_logging()
    assert logger.name == "trajsimbench"
    caplog.clear()
    with Timer() as timer:
        pass
    assert timer.elapsed_ns >= 0
    value, elapsed = timed_call(lambda x: x + 1, 2)
    assert value == 3 and elapsed >= 0
