from __future__ import annotations

import tempfile
import unittest

from trajsimbench.orchestration.runner import run_experiment
from trajsimbench.storage.artifacts import validate_run_directory


class TinyExecutionTests(unittest.TestCase):
    def test_tiny_cpu_run_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(
                {
                    "experiment_id": "test_tiny",
                    "output_root": directory,
                    "tiny_counts": {"database": 12, "queries": 3},
                    "methods": ["euclidean"],
                    "seed_list": [1],
                    "k_values": [1, 3],
                }
            )
            self.assertEqual(result.status, "complete")
            report = validate_run_directory(result.run_dir)
            self.assertTrue(report["valid"], report)


if __name__ == "__main__":
    unittest.main()
