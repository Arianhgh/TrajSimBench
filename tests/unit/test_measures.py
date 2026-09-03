import numpy as np
import pytest

from trajsimbench.measures import DistanceResult, get_measure, registry


def test_registry_contains_all_classical_measures():
    assert registry.names() == (
        "euclidean",
        "dtw",
        "hausdorff",
        "discrete_frechet",
        "lcss",
        "edr",
        "erp",
    )


@pytest.mark.parametrize("name", registry.names())
def test_identity_and_symmetry(name):
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    measure = get_measure(name)
    left = measure.distance(path, path).distance
    right = measure.distance(path[::-1], path).distance
    assert left == pytest.approx(0.0)
    assert right >= 0.0
    assert measure.distance(path, path[::-1]).distance == pytest.approx(right)


def test_euclidean_resamples_unequal_paths():
    a = np.array([[0.0, 0.0], [2.0, 0.0]])
    b = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert get_measure("euclidean", n_samples=7).distance(a, b).distance == pytest.approx(0.0)


def test_known_hausdorff_and_frechet_case():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [1.0, 1.0]])
    assert get_measure("hausdorff").distance(a, b).distance == pytest.approx(1.0)
    assert get_measure("discrete_frechet").distance(a, b).distance == pytest.approx(1.0)


def test_canonical_projected_columns_are_used():
    a = np.array([[100.0, 40.0, 0.0, 0.0, 1.0], [101.0, 40.0, 1.0, 0.0, 2.0]])
    b = np.array([[110.0, 50.0, 0.0, 0.0, 1.0], [111.0, 50.0, 1.0, 0.0, 2.0]])
    assert get_measure("dtw").distance(a, b).distance == pytest.approx(0.0)


def test_invalid_inputs_and_strict_config():
    with pytest.raises(ValueError):
        get_measure("dtw").distance(np.empty((0, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        get_measure("edr").distance(np.array([[np.nan, 0.0]]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        get_measure("euclidean", n_samples=0)
    with pytest.raises(ValueError):
        get_measure("lcss", epslion=1.0)


def test_lcss_time_delta_requires_timestamps():
    measure = get_measure("lcss", delta=1.0, delta_mode="time")
    with pytest.raises(ValueError, match="timestamps"):
        measure.distance(np.ones((2, 2)), np.ones((2, 2)))


def test_result_is_immutable_and_finite():
    result = DistanceResult(1.0, 1.0)
    with pytest.raises(AttributeError):
        result.distance = 2.0
    with pytest.raises(ValueError):
        DistanceResult(float("nan"), 0.0)
