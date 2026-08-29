from __future__ import annotations

import unittest
from dataclasses import dataclass

from tikitaka.retrieval.fusion import RRFConfig, reciprocal_rank_fusion, route_overlap


@dataclass(frozen=True)
class RouteHit:
    parent_asin: str
    rank: int
    score: float


class FusionTest(unittest.TestCase):
    def test_rrf_rewards_route_agreement_and_preserves_evidence(self) -> None:
        sparse = [RouteHit("A", 1, 9.0), RouteHit("B", 2, 8.0)]
        dense = [RouteHit("B", 1, 0.9), RouteHit("C", 2, 0.8)]
        fused = reciprocal_rank_fusion(
            sparse,
            dense,
            config=RRFConfig(k=60, candidate_limit=3),
            valid_ids={"A", "B", "C"},
        )
        self.assertEqual([item.parent_asin for item in fused], ["B", "A", "C"])
        self.assertEqual(fused[0].sparse_rank, 2)
        self.assertEqual(fused[0].dense_rank, 1)
        self.assertEqual(route_overlap(sparse, dense, depths=(1, 2)), {1: 0, 2: 1})

    def test_rrf_ties_are_stable_and_invalid_ids_are_removed(self) -> None:
        sparse = [RouteHit("B", 1, 4.0), RouteHit("A", 1, 3.0), RouteHit("X", 2, 9.0)]
        first = reciprocal_rank_fusion(sparse, (), valid_ids={"A", "B"})
        second = reciprocal_rank_fusion(sparse, (), valid_ids={"A", "B"})
        self.assertEqual(first, second)
        self.assertEqual([item.parent_asin for item in first], ["A", "B"])

    def test_rrf_rejects_nonfinite_scores_and_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "score must be finite"):
            reciprocal_rank_fusion([RouteHit("A", 1, float("nan"))], ())
        with self.assertRaisesRegex(ValueError, "weights must be finite"):
            RRFConfig(sparse_weight=float("inf"))


if __name__ == "__main__":
    unittest.main()
