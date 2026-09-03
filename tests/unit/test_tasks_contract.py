import unittest
from types import SimpleNamespace

import numpy as np

from trajsimbench.negatives import (
    NearbyShapeNegativeGenerator,
    RandomNegativeGenerator,
    SameODNegativeGenerator,
    TranslatedShapeNegativeGenerator,
)
from trajsimbench.tasks import (
    generate_diagnostic_triplets,
    generate_equivalence_tasks,
    generate_negative_task,
    generate_oracle_task,
)


def dataset():
    values = {}
    for index, offset in enumerate((0.0, 200.0, 500.0)):
        xy = np.column_stack(
            (np.arange(8, dtype=float) * 20.0 + offset, np.sin(np.arange(8, dtype=float)) * 20.0)
        )
        points = np.zeros((8, 5), dtype=float)
        points[:, 0] = -73.0 + xy[:, 0] / 111_000
        points[:, 1] = 45.0 + xy[:, 1] / 111_000
        points[:, 2:4] = xy
        points[:, 4] = np.arange(8) * 10.0
        values[str(index)] = SimpleNamespace(trajectory_id=str(index), points=points, metadata={})
    return values


class TaskContractTests(unittest.TestCase):
    def test_oracle_task_excludes_self_and_stable_ties(self):
        data = dataset()
        distances = {(q, c): float(int(q) + int(c)) for q in data for c in data if q != c}
        artifact = generate_oracle_task(
            data,
            query_ids=["0", "1"],
            database_ids=["0", "1", "2"],
            oracle_distances=distances,
            tie_tolerance=0.1,
            seed=4,
        )
        self.assertEqual(len(artifact.records), 2)
        self.assertNotIn("0", artifact.records[0]["candidate_ids"])
        self.assertTrue(artifact.content_hash)
        self.assertEqual(
            artifact.content_hash,
            generate_oracle_task(
                data,
                query_ids=["0", "1"],
                database_ids=["0", "1", "2"],
                oracle_distances=distances,
                tie_tolerance=0.1,
                seed=4,
            ).content_hash,
        )

    def test_equivalence_and_diagnostics_are_deterministic(self):
        data = dataset()
        first = generate_equivalence_tasks(
            data, source_ids=["0", "1"], perturbations=[("gps_noise", 5.0)], seed=11
        )
        second = generate_equivalence_tasks(
            data, source_ids=["0", "1"], perturbations=[("gps_noise", 5.0)], seed=11
        )
        self.assertEqual(first.content_hash, second.content_hash)
        diagnostic = generate_diagnostic_triplets(
            data, family="translated_vs_nearby", notion="geometric_shape", count=2, seed=2
        )
        self.assertEqual(len(diagnostic.records), 2)
        self.assertEqual(diagnostic.records[0]["expected_order"], "a_closer")
        absolute = generate_diagnostic_triplets(
            data, family="translated_vs_nearby", notion="absolute_geographic_route", count=1, seed=2
        )
        self.assertEqual(absolute.records[0]["expected_order"], "b_closer")

    def test_negative_generators_report_bounds_and_no_self_match(self):
        data = dataset()
        for generator in (
            RandomNegativeGenerator(),
            NearbyShapeNegativeGenerator(),
            TranslatedShapeNegativeGenerator(),
            SameODNegativeGenerator(),
        ):
            result = generator.generate(
                "0",
                ["0", "1", "2"],
                dataset=data,
                count=1,
                seed=3,
                config={"max_candidates_examined": 1},
            )
            self.assertLessEqual(result.report.attempted, 1)
            self.assertNotIn("0", result.report.to_dict()["config"].get("candidate_ids", []))
            self.assertEqual(result.report.accepted, len(result.candidates))

    def test_negative_task_persists_construction_reports(self):
        artifact = generate_negative_task(
            dataset(),
            generator="random",
            query_ids=["0", "1"],
            database_ids=["0", "1", "2"],
            seed=8,
        )
        self.assertEqual(len(artifact.records), 2)
        self.assertIn("construction_reports", artifact.metadata)
        self.assertTrue(artifact.content_hash)


if __name__ == "__main__":
    unittest.main()
