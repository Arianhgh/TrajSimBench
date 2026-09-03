import unittest
from types import SimpleNamespace

import numpy as np

from trajsimbench.perturbations import PERTURBATION_REGISTRY


def make_view(n=12):
    xy = np.column_stack(
        (np.arange(n, dtype=float) * 10.0, np.sin(np.arange(n, dtype=float)) * 10.0)
    )
    points = np.zeros((n, 5), dtype=np.float64)
    points[:, 0] = -73.0 + xy[:, 0] / 111_000.0
    points[:, 1] = 45.0 + xy[:, 1] / 111_000.0
    points[:, 2:4] = xy
    points[:, 4] = np.arange(n, dtype=float) * 10.0
    return SimpleNamespace(trajectory_id="source-1", points=points, metadata={})


class PerturbationContractTests(unittest.TestCase):
    def test_all_free_space_v1_transforms_are_registered_and_non_mutating(self):
        source = make_view()
        original = source.points.copy()
        severities = {
            "gps_noise": 5.0,
            "gps_drift": 5.0,
            "random_point_loss": 0.75,
            "contiguous_outage": 0.10,
            "sampling_reduction": 0.75,
            "spatial_quantization": 10.0,
            "temporal_jitter": 1.0,
            "speed_distortion": 2.0,
            "truncation": 0.10,
            "reversal": {"timestamp_policy": "rebase"},
            "spatial_translation": {"magnitude_m": 100.0, "bearing_rad": 0.5},
            "free_space_detour": 0.10,
        }
        for name, severity in severities.items():
            result = PERTURBATION_REGISTRY.apply(name, source, severity=severity, seed=42)
            self.assertTrue(result.generated, (name, result.reason))
            self.assertFalse(result.points.flags.writeable)
            self.assertIsNot(result.points, source.points)
            self.assertEqual(result.provenance.input_hash, result.provenance["input_hash"])
            result.validate(metadata=source.metadata)
        np.testing.assert_array_equal(source.points, original)

    def test_seed_and_regeneration_are_byte_deterministic(self):
        source = make_view()
        perturbation = PERTURBATION_REGISTRY.get("gps_noise")
        first = perturbation.apply(source, severity=25.0, seed=123)
        second = perturbation.apply(source, severity=25.0, seed=123)
        self.assertEqual(first.provenance.output_hash, second.provenance.output_hash)
        np.testing.assert_array_equal(first.points, second.points)
        regenerated = perturbation.regenerate(source, first.provenance)
        np.testing.assert_array_equal(first.points, regenerated.points)
        different = perturbation.apply(source, severity=25.0, seed=124)
        self.assertNotEqual(first.provenance.output_hash, different.provenance.output_hash)

    def test_rng_only_provenance_can_regenerate(self):
        source = make_view()
        perturbation = PERTURBATION_REGISTRY.get("temporal_jitter")
        first = perturbation.apply(source, severity=1.0, rng=np.random.default_rng(3))
        regenerated = perturbation.regenerate(source, first.provenance)
        np.testing.assert_array_equal(first.points, regenerated.points)

    def test_short_trajectory_and_disabled_road_network_are_typed_rejections(self):
        source = make_view(2)
        short = PERTURBATION_REGISTRY.apply("free_space_detour", source, severity=0.2, seed=1)
        self.assertEqual(short.status, "not_generated")
        road = PERTURBATION_REGISTRY.apply("road_network_detour", make_view(), severity=0.2, seed=1)
        self.assertEqual(road.status, "not_generated")
        self.assertIn("disabled", road.reason)


if __name__ == "__main__":
    unittest.main()
