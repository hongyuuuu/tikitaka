from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.retrieval_fakes import SemanticFakeEmbedder
from tikitaka.config import RuntimeRoutingConfig
from tikitaka.contracts import (
    Attribute,
    Candidate,
    EvidenceOutcome,
    InferredMode,
    ProductEvidence,
    ProfileBias,
    RoutePolicy,
    Retriever,
    SearchPlan,
)
from tikitaka.retrieval.adapters import ContractRetrieverAdapter
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact, load_dense_index
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.retriever import SparseStructuredRetriever


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class RetrievalContractAdapterTest(unittest.TestCase):
    def test_sparse_search_satisfies_canonical_retriever_protocol(self) -> None:
        catalog = load_catalog(FIXTURE)
        plan = SearchPlan(
            text_query="comfortable walking shoes",
            must_terms=(),
            should_terms=(),
            exclude_terms=(),
            filters={},
            attribute_values={Attribute.CATEGORY: ("shoes",)},
            mode=InferredMode.BUYING,
            intent_version=1,
            revalidation_flags=frozenset(),
            no_preference=frozenset(),
            profile_bias=ProfileBias(),
            route_policy=RoutePolicy.SPARSE,
            embedding_route_id=None,
            index_id=None,
        )
        with SparseStructuredRetriever(catalog) as retriever:
            self.assertIsInstance(retriever, Retriever)
            candidates = retriever.search(plan, 5)

        self.assertTrue(candidates)
        self.assertTrue(all(isinstance(item, Candidate) for item in candidates))
        self.assertEqual(len(candidates), len({item.parent_asin for item in candidates}))

    def test_hybrid_search_returns_canonical_candidate_and_evidence_contracts(self) -> None:
        catalog = load_catalog(FIXTURE)
        embedder = SemanticFakeEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dense_artifact(
                catalog,
                embedder,
                directory,
                embedding_provider="fixture",
                embedding_model="semantic-keywords-v1",
            )
            index = load_dense_index(directory, catalog)
            RuntimeRoutingConfig(
                retrieval_policy=RoutePolicy.HYBRID,
                embedding_route_id=manifest.route_id,
                index_id=manifest.index_id,
            ).validate_index(manifest)
            plan = SearchPlan(
                text_query="waterproof hiking shoes",
                must_terms=(),
                should_terms=("walking",),
                exclude_terms=(),
                filters={"budget": {"max": 80}},
                attribute_values={
                    Attribute.CATEGORY: ("shoes",),
                    Attribute.BUDGET: (80,),
                },
                mode=InferredMode.BUYING,
                intent_version=1,
                revalidation_flags=frozenset(),
                no_preference=frozenset(),
                profile_bias=ProfileBias(),
                route_policy=RoutePolicy.HYBRID,
                embedding_route_id=manifest.route_id,
                index_id=manifest.index_id,
            )
            with HybridRetriever(
                catalog,
                dense_index=index,
                query_embedder=embedder,
            ) as retriever:
                self.assertIsInstance(retriever, Retriever)
                candidates = retriever.search(plan, 5)
                wrapped = ContractRetrieverAdapter(retriever).search(plan, 5)

        self.assertTrue(candidates)
        self.assertEqual(candidates, wrapped)
        self.assertTrue(all(isinstance(item, Candidate) for item in candidates))
        self.assertTrue(all(item.parent_asin in catalog for item in candidates))
        evidence = candidates[0].product_evidence
        self.assertIsInstance(evidence, ProductEvidence)
        self.assertEqual(set(evidence.constraint_outcomes), set(Attribute))
        self.assertEqual(set(evidence.attribute_values), set(Attribute))
        self.assertIn(
            evidence.constraint_outcomes[Attribute.BUDGET],
            {EvidenceOutcome.MATCH, EvidenceOutcome.UNKNOWN},
        )
        self.assertEqual(evidence.route_details["dense_index_id"], manifest.index_id)


if __name__ == "__main__":
    unittest.main()
