from __future__ import annotations

import unittest

from trajsimbench.evaluation.retrieval import evaluate_retrieval, ndcg_at_k, recall_at_k


class RetrievalMetricTests(unittest.TestCase):
    def test_binary_and_graded_metrics(self) -> None:
        ranked = ["b", "a", "c"]
        relevance = {"a": 1.0, "b": 0.0, "c": 1.0}
        self.assertAlmostEqual(recall_at_k(ranked, relevance, 1), 0.0)
        self.assertAlmostEqual(recall_at_k(ranked, relevance, 2), 0.5)
        self.assertLess(ndcg_at_k(ranked, relevance, 1), 1.0)

    def test_empty_policy_is_explicit(self) -> None:
        rows = evaluate_retrieval({"q": ["a"]}, {"q": {"a": 0.0}}, ks=[1], empty_policy="skip")
        self.assertTrue(all(not row["valid"] for row in rows))


if __name__ == "__main__":
    unittest.main()
