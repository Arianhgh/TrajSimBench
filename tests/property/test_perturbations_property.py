import unittest
from types import SimpleNamespace

import numpy as np

from trajsimbench.perturbations import PERTURBATION_REGISTRY


class PerturbationPropertyTests(unittest.TestCase):
    def test_spatial_outputs_remain_finite_and_timestamps_monotonic(self):
        for seed in range(8):
            n = 4 + seed
            points = np.zeros((n, 5), dtype=float)
            points[:, 0] = -73.0 + np.arange(n) / 100_000
            points[:, 1] = 45.0
            points[:, 2] = np.arange(n) * 25.0
            points[:, 3] = np.arange(n) * 5.0
            points[:, 4] = np.arange(n) * 2.0
            source = SimpleNamespace(trajectory_id=f"p-{seed}", points=points, metadata={})
            for name, severity in (
                ("gps_noise", 5.0),
                ("gps_drift", 5.0),
                ("spatial_quantization", 10.0),
                ("spatial_translation", 100.0),
                ("free_space_detour", 0.1),
            ):
                result = PERTURBATION_REGISTRY.apply(name, source, severity=severity, seed=seed)
                if result.generated:
                    self.assertTrue(np.isfinite(result.points[:, 2:4]).all())
                    self.assertTrue(np.all(np.diff(result.points[:, 4]) >= 0))

    def test_zero_noise_and_zero_detour_are_identity_values(self):
        points = np.column_stack((np.arange(6, dtype=float), np.arange(6, dtype=float) ** 2))
        source = SimpleNamespace(trajectory_id="identity", points=points, metadata={})
        for name in ("gps_noise", "gps_drift", "free_space_detour"):
            result = PERTURBATION_REGISTRY.apply(name, source, severity=0.0, seed=7)
            self.assertTrue(result.generated)
            np.testing.assert_array_equal(result.points, points)


if __name__ == "__main__":
    unittest.main()
