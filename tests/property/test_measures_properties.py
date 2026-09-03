"""Deterministic property checks for every classical method."""

import numpy as np
import pytest

from trajsimbench.measures import create_measure, registry


@pytest.mark.parametrize("name", registry.names())
def test_finite_nonnegative_symmetric_and_identity(name):
    rng = np.random.default_rng(1000 + list(registry.names()).index(name))
    measure = create_measure(name)
    for _ in range(12):
        a = rng.normal(size=(int(rng.integers(1, 7)), 2)).cumsum(axis=0)
        b = rng.normal(size=(int(rng.integers(1, 7)), 2)).cumsum(axis=0)
        left = measure.distance(a, b)
        right = measure.distance(b, a)
        identity = measure.distance(a, a)
        assert np.isfinite(left.distance)
        assert left.distance >= 0
        assert left.distance == pytest.approx(right.distance, abs=1e-12)
        assert identity.distance == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", registry.names())
def test_repeated_points_and_singletons_remain_finite(name):
    measure = create_measure(name)
    repeated = np.array([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
    singleton = np.array([[2.0, -1.0]])
    result = measure.distance(repeated, singleton)
    assert np.isfinite(result.distance)
    assert result.distance >= 0


@pytest.mark.parametrize("name", registry.names())
def test_nan_and_empty_inputs_are_rejected(name):
    measure = create_measure(name)
    with pytest.raises(ValueError):
        measure.distance(np.empty((0, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        measure.distance(np.array([[np.nan, 0.0]]), np.ones((1, 2)))


def test_extreme_thresholds_have_documented_bounds():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[100.0, 0.0], [110.0, 0.0], [120.0, 0.0]])
    assert create_measure("lcss", epsilon=np.inf).distance(a, b).distance == pytest.approx(0.0)
    assert create_measure("edr", epsilon=np.inf).distance(a, b).distance == pytest.approx(1 / 3)
    assert create_measure("lcss", epsilon=0.0).distance(a, b).distance == pytest.approx(1.0)


def test_time_constrained_lcss_requires_timestamps_and_accepts_them():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [1.0, 0.0]])
    measure = create_measure("lcss", delta=0.5, delta_mode="time")
    with pytest.raises(ValueError, match="timestamps"):
        measure.distance(a, b)
    timed_a = np.column_stack((a, [0.0, 1.0]))
    timed_b = np.column_stack((b, [0.0, 1.4]))
    assert measure.distance(timed_a, timed_b).distance == pytest.approx(0.0)
