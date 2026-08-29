from __future__ import annotations

import unittest
from pathlib import Path

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.request import RetrievalConstraint, RetrievalRequest
from tikitaka.retrieval.retriever import RetrievalConfig, SparseStructuredRetriever


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class HybridRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(FIXTURE)
        cls.retriever = SparseStructuredRetriever(cls.catalog)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.retriever.close()

    def test_hard_budget_filters_known_conflict_but_keeps_unknown(self) -> None:
        request = RetrievalRequest(
            text_query="comfortable shoes",
            constraints=(
                RetrievalConstraint(
                    attribute="budget", values=(60,), strength="hard", operator="lte"
                ),
            ),
            mode="buying",
        )
        result = self.retriever.retrieve(request, limit=10)
        identifiers = [hit.parent_asin for hit in result.hits]
        self.assertIn("B_RUN", identifiers)
        self.assertIn("C_WALK_UNKNOWN", identifiers)
        self.assertIn("F_FASHION", identifiers)
        self.assertNotIn("A_HIKE", identifiers)
        self.assertGreaterEqual(result.diagnostics.hard_filtered_candidates, 1)

    def test_explicit_material_exclusion_filters_only_positive_evidence(self) -> None:
        request = RetrievalRequest(
            text_query="hiking boot",
            constraints=(
                RetrievalConstraint(
                    attribute="material",
                    values=("leather",),
                    polarity="exclude",
                    strength="hard",
                ),
            ),
        )
        identifiers = [hit.parent_asin for hit in self.retriever.search_hits(request, 10)]
        self.assertNotIn("A_HIKE", identifiers)
        self.assertIn("E_CANVAS_BOOT", identifiers)

    def test_boost_only_ablation_retains_known_hard_contradictions(self) -> None:
        request = RetrievalRequest(
            text_query="comfortable shoes",
            constraints=(
                RetrievalConstraint(
                    attribute="budget", values=(60,), strength="hard", operator="lte"
                ),
            ),
        )
        with SparseStructuredRetriever(
            self.catalog,
            retrieval_config=RetrievalConfig(hard_filtering=False),
        ) as retriever:
            result = retriever.retrieve(request, limit=10)

        identifiers = [hit.parent_asin for hit in result.hits]
        self.assertIn("A_HIKE", identifiers)
        self.assertEqual(result.diagnostics.hard_filtered_candidates, 0)

    def test_profile_weight_zero_is_provably_inert(self) -> None:
        without_profile = RetrievalRequest(text_query="shoes")
        zero_profile = RetrievalRequest(
            text_query="shoes", profile_terms=("blue", "travel"), profile_weight=0.0
        )
        first = self.retriever.search_hits(without_profile, 10)
        second = self.retriever.search_hits(zero_profile, 10)
        self.assertEqual(
            [(hit.parent_asin, hit.fused_score) for hit in first],
            [(hit.parent_asin, hit.fused_score) for hit in second],
        )

    def test_positive_constraints_supply_recall_terms_without_raw_query(self) -> None:
        request = RetrievalRequest(
            text_query="",
            constraints=(RetrievalConstraint("category", ("shoes",), strength="soft"),),
        )
        identifiers = [hit.parent_asin for hit in self.retriever.search_hits(request, 10)]
        self.assertIn("A_HIKE", identifiers)
        self.assertNotIn("G_TOTE", identifiers)

    def test_profile_prior_cannot_override_explicit_dialogue_match(self) -> None:
        request = RetrievalRequest(
            text_query="shoes",
            constraints=(RetrievalConstraint("material", ("cotton",), strength="soft"),),
            profile_terms=("leather",),
            profile_weight=1.0,
        )
        hits = self.retriever.search_hits(request, 10)
        self.assertEqual(hits[0].parent_asin, "B_RUN")
        self.assertTrue(all(hit.profile_contribution == 0.0 for hit in hits))

    def test_results_are_valid_unique_evidence_bearing_and_stable(self) -> None:
        request = RetrievalRequest(text_query="travel comfort walking", mode="browsing")
        first = self.retriever.retrieve(request, limit=5)
        second = self.retriever.retrieve(request, limit=5)
        first_ids = [hit.parent_asin for hit in first.hits]
        self.assertEqual(first_ids, [hit.parent_asin for hit in second.hits])
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertTrue(set(first_ids).issubset(self.catalog.ids))
        self.assertTrue(all(hit.matched_fields for hit in first.hits))
        self.assertEqual(first.diagnostics.route, "sparse_structured")


if __name__ == "__main__":
    unittest.main()
