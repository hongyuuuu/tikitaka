from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.retrieval_fakes import FailingQueryEmbedder, SemanticFakeEmbedder
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact, load_dense_index
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.request import RetrievalConstraint, RetrievalRequest


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class HybridDenseRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(FIXTURE)
        cls.directory = tempfile.TemporaryDirectory()
        cls.embedder = SemanticFakeEmbedder()
        cls.manifest = build_dense_artifact(
            cls.catalog,
            cls.embedder,
            cls.directory.name,
            embedding_provider="fixture",
            embedding_model="semantic-keywords-v1",
            batch_size=3,
        )
        cls.index = load_dense_index(cls.directory.name, cls.catalog)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_dense_and_hybrid_routes_are_pinnable_and_evidence_bearing(self) -> None:
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=self.embedder,
        ) as retriever:
            dense = retriever.retrieve(
                RetrievalRequest(
                    text_query="rainproof mountain footwear for a trip",
                    route_policy="dense",
                    embedding_route_id=self.manifest.route_id,
                    index_id=self.manifest.index_id,
                ),
                limit=5,
            )
            hybrid = retriever.retrieve(
                RetrievalRequest(
                    text_query="waterproof hiking boots",
                    route_policy="hybrid",
                    embedding_route_id=self.manifest.route_id,
                    index_id=self.manifest.index_id,
                ),
                limit=5,
            )

        self.assertEqual(dense.diagnostics.executed_route, "dense")
        self.assertIn("A_HIKE", [hit.parent_asin for hit in dense.hits[:2]])
        self.assertTrue(all(hit.dense_rank is not None for hit in dense.hits))
        self.assertEqual(hybrid.diagnostics.executed_route, "hybrid")
        self.assertTrue(any(hit.sparse_rank and hit.dense_rank for hit in hybrid.hits))
        self.assertEqual(
            hybrid.diagnostics.manifest_ids["dense"],
            self.manifest.index_id,
        )

    def test_wrong_pin_and_query_failure_degrade_to_sparse(self) -> None:
        mismatch = RetrievalRequest(
            text_query="waterproof hiking boots",
            route_policy="hybrid",
            embedding_route_id="wrong-route",
            index_id="wrong-index",
        )
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=self.embedder,
        ) as retriever:
            mismatch_result = retriever.retrieve(mismatch, limit=5)
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=FailingQueryEmbedder(),
        ) as retriever:
            failure_result = retriever.retrieve(
                RetrievalRequest(
                    text_query="waterproof hiking boots",
                    route_policy="dense",
                    embedding_route_id=self.manifest.route_id,
                    index_id=self.manifest.index_id,
                ),
                limit=5,
            )

        self.assertEqual(mismatch_result.diagnostics.executed_route, "sparse_fallback")
        self.assertIn("embedding_route_mismatch", mismatch_result.diagnostics.route_failures)
        self.assertTrue(all(hit.dense_rank is None for hit in mismatch_result.hits))
        self.assertEqual(failure_result.diagnostics.executed_route, "sparse_fallback")
        self.assertIn("dense_query_failed", failure_result.diagnostics.route_failures)
        self.assertTrue(failure_result.hits)

    def test_constraints_and_generality_diagnostics_survive_fusion(self) -> None:
        request = RetrievalRequest(
            text_query="comfortable travel walking shoes",
            constraints=(
                RetrievalConstraint("category", ("shoes",), strength="soft"),
                RetrievalConstraint("budget", (60,), strength="hard", operator="lte"),
            ),
            mode="buying",
            route_policy="hybrid",
            embedding_route_id=self.manifest.route_id,
            index_id=self.manifest.index_id,
        )
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=self.embedder,
        ) as retriever:
            result = retriever.retrieve(request, limit=7)

        identifiers = [hit.parent_asin for hit in result.hits]
        self.assertIn("C_WALK_UNKNOWN", identifiers)
        self.assertNotIn("A_HIKE", identifiers)
        self.assertGreaterEqual(result.diagnostics.hard_filtered_candidates, 1)
        self.assertIn("budget", result.diagnostics.missing_attribute_rates)
        self.assertIn("budget", result.diagnostics.constraint_outcome_counts)
        self.assertGreater(result.diagnostics.effective_candidate_count, 0.0)

    def test_no_dense_artifact_completes_through_deterministic_fallback(self) -> None:
        with HybridRetriever(self.catalog) as retriever:
            result = retriever.retrieve(
                RetrievalRequest(
                    text_query="waterproof hiking boots",
                    route_policy="auto",
                ),
                limit=5,
            )
        self.assertTrue(result.hits)
        self.assertEqual(result.diagnostics.executed_route, "sparse_fallback")
        self.assertEqual(result.diagnostics.route_failures, ("dense_route_unavailable",))
        self.assertTrue(all(hit.parent_asin in self.catalog for hit in result.hits))

    def test_hybrid_order_is_stable_unique_valid_and_limit_bounded(self) -> None:
        request = RetrievalRequest(
            text_query="waterproof hiking travel shoes",
            route_policy="hybrid",
            embedding_route_id=self.manifest.route_id,
            index_id=self.manifest.index_id,
        )
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=self.embedder,
        ) as retriever:
            first = retriever.search_hits(request, 4)
            second = retriever.search_hits(request, 4)
        first_signature = [
            (hit.parent_asin, hit.sparse_rank, hit.dense_rank, hit.fused_score)
            for hit in first
        ]
        second_signature = [
            (hit.parent_asin, hit.sparse_rank, hit.dense_rank, hit.fused_score)
            for hit in second
        ]
        self.assertEqual(first_signature, second_signature)
        self.assertLessEqual(len(first), 4)
        self.assertEqual(len({hit.parent_asin for hit in first}), len(first))
        self.assertTrue(all(hit.parent_asin in self.catalog for hit in first))

    def test_new_intent_uses_only_reduced_current_state_without_hidden_history(self) -> None:
        old_intent = RetrievalRequest(
            text_query="leather waterproof hiking boots",
            constraints=(RetrievalConstraint("material", ("leather",)),),
            intent_version=1,
            route_policy="hybrid",
            embedding_route_id=self.manifest.route_id,
            index_id=self.manifest.index_id,
        )
        new_intent = RetrievalRequest(
            text_query="cotton running shoes",
            constraints=(RetrievalConstraint("material", ("cotton",)),),
            intent_version=2,
            route_policy="hybrid",
            embedding_route_id=self.manifest.route_id,
            index_id=self.manifest.index_id,
        )
        with HybridRetriever(
            self.catalog,
            dense_index=self.index,
            query_embedder=self.embedder,
        ) as retriever:
            old = retriever.search_hits(old_intent, 5)
            changed = retriever.search_hits(new_intent, 5)
            repeated = retriever.search_hits(new_intent, 5)

        self.assertEqual(changed[0].parent_asin, "B_RUN")
        self.assertNotEqual(old[0].parent_asin, changed[0].parent_asin)
        self.assertEqual(
            [(hit.parent_asin, hit.fused_score) for hit in changed],
            [(hit.parent_asin, hit.fused_score) for hit in repeated],
        )


if __name__ == "__main__":
    unittest.main()
