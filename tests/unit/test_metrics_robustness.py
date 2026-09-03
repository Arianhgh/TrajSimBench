from __future__ import annotations

import unittest

from trajsimbench.evaluation.robustness import (
    monotonicity_violation_rate,
    robustness_auc,
    robustness_curve,
)


class RobustnessMetricTests(unittest.TestCase):
    def test_auc_and_pairwise_mvr(self) -> None:
        curve = robustness_curve([0, 0.5, 1], 1.0, [1.0, 0.8, 0.6], mode="quality")
        self.assertAlmostEqual(robustness_auc(curve), 0.8)
        report = monotonicity_violation_rate({"s": [0.0, 1.0, 0.5]})
        self.assertAlmostEqual(report["rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
