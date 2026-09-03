from __future__ import annotations

import unittest

from trajsimbench.retrieval.ranking import (
    kendall_tau_b,
    rank_biased_overlap,
    spearman_rho,
    top_k_jaccard,
)


class RankingTests(unittest.TestCase):
    def test_perfect_and_reversed_order(self) -> None:
        self.assertAlmostEqual(kendall_tau_b(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertAlmostEqual(spearman_rho(["a", "b", "c"], ["c", "b", "a"]), -1.0)

    def test_set_and_persistence_agreement(self) -> None:
        self.assertAlmostEqual(top_k_jaccard(["a", "b"], ["b", "a"], 2), 1.0)
        self.assertGreater(rank_biased_overlap(["a", "b"], ["a", "c"], persistence=0.9), 0.0)


if __name__ == "__main__":
    unittest.main()
