from __future__ import annotations

import unittest


class DashboardImportTests(unittest.TestCase):
    def test_pages_are_streamlit_safe(self) -> None:
        from trajsimbench.dashboard.pages import (
            builder,
            counterfactual,
            dataset,
            disagreement,
            efficiency,
            fingerprints,
            pair,
            robustness,
        )

        self.assertTrue(callable(builder.render))
        self.assertTrue(callable(dataset.render))
        self.assertTrue(callable(pair.render))
        self.assertTrue(callable(counterfactual.render))
        self.assertTrue(callable(disagreement.render))
        self.assertTrue(callable(fingerprints.render))
        self.assertTrue(callable(robustness.render))
        self.assertTrue(callable(efficiency.render))


if __name__ == "__main__":
    unittest.main()
