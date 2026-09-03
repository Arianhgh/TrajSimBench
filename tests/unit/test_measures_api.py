"""Unit tests for the public typed measure API and strict configurations."""

import numpy as np
import pytest

from trajsimbench.measures import (
    DistanceResult,
    MeasureCapabilities,
    MeasureCapabilityError,
    TrajectoryView,
    create_measure,
    registry,
)
from trajsimbench.measures.config import ConfigValidationError
from trajsimbench.measures.timing import time_pairwise


def test_capabilities_and_result_are_frozen_contracts():
    capabilities = MeasureCapabilities(supports_batch=True)
    assert capabilities.supports_batch
    with pytest.raises((AttributeError, TypeError)):
        capabilities.supports_batch = False

    result = DistanceResult(1, 1)
    assert result.distance == 1.0
    assert result.runtime_ns is None
    with pytest.raises((AttributeError, TypeError)):
        result.distance = 2


def test_trajectory_view_preserves_non_owning_points():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    view = TrajectoryView("example", points, {"source": "unit"})
    assert view.points is points
    assert view.trajectory_id == "example"
    assert view.metadata["source"] == "unit"


def test_registry_is_strict_and_supports_proposal_iteration():
    assert registry.names() == (
        "euclidean",
        "dtw",
        "hausdorff",
        "discrete_frechet",
        "lcss",
        "edr",
        "erp",
    )
    assert [measure.name for measure in registry] == list(registry.names())
    assert all(measure.capabilities.supports_batch for measure in registry)
    with pytest.raises(KeyError, match="unknown measure"):
        create_measure("not_a_measure")
    with pytest.raises(ConfigValidationError, match="unknown field"):
        create_measure("euclidean", n_sampels=5)
    with pytest.raises(ConfigValidationError):
        create_measure("dtw", normalization="average")


def test_pairwise_and_timing_harness_are_deterministic_in_shape():
    query = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidates = [query, np.array([[0.0, 1.0], [1.0, 1.0]])]
    measure = create_measure("euclidean", n_samples=5)
    values = measure.pairwise(query, candidates)
    assert values.shape == (2,)
    assert values[0] == pytest.approx(0.0)
    timing = time_pairwise(measure, query, candidates, warmup=1, repetitions=2)
    assert timing.candidate_count == 2
    assert len(timing.samples_ns) == 2
    assert timing.median_ns >= 0
    assert timing.p95_ns >= timing.median_ns
    assert timing.as_dict()["measure"] == "euclidean"


def test_capability_error_is_typed():
    with pytest.raises(MeasureCapabilityError):
        create_measure("dtw").encode([])


@pytest.mark.parametrize("name", registry.names())
def test_canonical_projection_ignores_lon_lat_and_timestamp(name):
    a = np.array([[120.0, 35.0, 0.0, 0.0, 10.0], [121.0, 36.0, 2.0, 0.0, 20.0]])
    b = np.array([[-80.0, 5.0, 0.0, 0.0, 100.0], [-81.0, 6.0, 2.0, 0.0, 200.0]])
    assert create_measure(name).distance(a, b).distance == pytest.approx(0.0)
