"""Independent slow references for the seven frozen classical definitions."""

import numpy as np
import pytest

from trajsimbench.measures import create_measure


def ref_resample(points, count):
    if count == 1:
        return points[:1]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if cumulative[-1] == 0:
        return np.repeat(points[:1], count, axis=0)
    unique, indices = np.unique(cumulative, return_index=True)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([np.interp(targets, unique, points[indices, axis]) for axis in range(2)])


def ref_euclidean(a, b, count):
    return float(np.linalg.norm(ref_resample(a, count) - ref_resample(b, count), axis=1).mean())


def ref_dtw(a, b):
    dp = np.full((len(a) + 1, len(b) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1])


def ref_hausdorff(a, b):
    matrix = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(matrix.min(axis=1).max(), matrix.min(axis=0).max()))


def ref_frechet(a, b):
    costs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    dp = np.empty_like(costs)
    dp[0, 0] = costs[0, 0]
    for i in range(1, len(a)):
        dp[i, 0] = max(dp[i - 1, 0], costs[i, 0])
    for j in range(1, len(b)):
        dp[0, j] = max(dp[0, j - 1], costs[0, j])
    for i in range(1, len(a)):
        for j in range(1, len(b)):
            dp[i, j] = max(costs[i, j], min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]))
    return float(dp[-1, -1])


def ref_lcss(a, b, epsilon):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if np.linalg.norm(a[i - 1] - b[j - 1]) <= epsilon:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return 1.0 - dp[-1, -1] / min(len(a), len(b))


def ref_edr(a, b, epsilon):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            sub = int(np.linalg.norm(a[i - 1] - b[j - 1]) > epsilon)
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + sub)
    return dp[-1, -1] / max(len(a), len(b))


def ref_erp(a, b, gap):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=float)
    dp[1:, 0] = np.cumsum(np.linalg.norm(a - gap, axis=1))
    dp[0, 1:] = np.cumsum(np.linalg.norm(b - gap, axis=1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = min(
                dp[i - 1, j - 1] + np.linalg.norm(a[i - 1] - b[j - 1]),
                dp[i - 1, j] + np.linalg.norm(a[i - 1] - gap),
                dp[i, j - 1] + np.linalg.norm(b[j - 1] - gap),
            )
    return float(dp[-1, -1])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("euclidean", lambda a, b: ref_euclidean(a, b, 7)),
        ("dtw", ref_dtw),
        ("hausdorff", ref_hausdorff),
        ("discrete_frechet", ref_frechet),
        ("lcss", lambda a, b: ref_lcss(a, b, 0.4)),
        ("edr", lambda a, b: ref_edr(a, b, 0.4)),
        ("erp", lambda a, b: ref_erp(a, b, np.array([0.25, -0.5]))),
    ],
)
def test_random_small_paths_match_independent_reference(name, expected):
    rng = np.random.default_rng(20260903)
    a = rng.normal(size=(4, 2)).cumsum(axis=0)
    b = rng.normal(size=(3, 2)).cumsum(axis=0)
    configs = {
        "euclidean": {"n_samples": 7},
        "lcss": {"epsilon": 0.4},
        "edr": {"epsilon": 0.4},
        "erp": {"gap_point": (0.25, -0.5)},
    }
    actual = create_measure(name, **configs.get(name, {})).distance(a, b)
    assert actual.distance == pytest.approx(expected(a, b), rel=1e-12, abs=1e-12)
    if name == "lcss":
        assert actual.raw_score == pytest.approx((1.0 - expected(a, b)) * min(len(a), len(b)))
    elif name == "edr":
        assert actual.raw_score == pytest.approx(expected(a, b) * max(len(a), len(b)))
    else:
        assert actual.raw_score == pytest.approx(expected(a, b), rel=1e-12, abs=1e-12)


def test_reference_normalizations_are_explicit():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    raw = create_measure("dtw", normalization="none").distance(a, b)
    by_inputs = create_measure("dtw", normalization="max_input_length").distance(a, b)
    by_path = create_measure("dtw", normalization="path_length").distance(a, b)
    assert raw.raw_score == pytest.approx(0.0)
    assert by_inputs.distance == pytest.approx(0.0)
    assert by_path.distance == pytest.approx(0.0)
    assert by_path.details["normalization_denominator"] >= 1
