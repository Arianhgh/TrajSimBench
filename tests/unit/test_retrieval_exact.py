from __future__ import annotations

import unittest

from trajsimbench.retrieval.exact import exact_top_k, pairwise_distances


class ExactRetrievalTests(unittest.TestCase):
    def test_chunked_matches_full_and_tie_breaks_by_id(self) -> None:
        candidates = [0.0, 1.0, 1.0, 2.0]
        full = exact_top_k(
            0.0,
            candidates,
            k=4,
            candidate_ids=["z", "b", "a", "c"],
            distance_fn=lambda q, x: abs(q - x),
            chunk_size=4,
        )
        chunked = exact_top_k(
            0.0,
            candidates,
            k=4,
            candidate_ids=["z", "b", "a", "c"],
            distance_fn=lambda q, x: abs(q - x),
            chunk_size=2,
        )
        self.assertEqual(full.candidate_ids.tolist(), chunked.candidate_ids.tolist())
        self.assertEqual(full.candidate_ids.tolist(), ["z", "a", "b", "c"])

    def test_exclusion_and_pairwise_candidate_order(self) -> None:
        ids = ["a", "b", "c"]
        result = exact_top_k(
            0.0,
            [0.0, 2.0, 1.0],
            k=2,
            candidate_ids=ids,
            distance_fn=lambda q, x: abs(q - x),
            exclude_ids={"a"},
        )
        self.assertEqual(result.ids.tolist(), ["c", "b"])
        distances, _ = pairwise_distances(
            0.0, [0.0, 2.0, 1.0], distance_fn=lambda q, x: abs(q - x), chunk_size=2
        )
        self.assertEqual(distances.tolist(), [0.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
