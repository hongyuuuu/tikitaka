from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from tests.retrieval_fakes import SemanticFakeEmbedder
from tikitaka.retrieval.adapters import ContractRetrieverAdapter
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact, load_dense_index
from tikitaka.retrieval.hybrid import HybridRetriever


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


@dataclass(frozen=True)
class FrozenEvidenceShape:
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]
    constraint_outcomes: Mapping[str, str]
    attribute_values: Mapping[str, tuple[object, ...]]
    evidence_reliability: Mapping[str, float]
    unknown_fields: tuple[str, ...]
    route_details: Mapping[str, object]
    profile_contribution: float = 0.0


@dataclass(frozen=True)
class FrozenCandidateShape:
    parent_asin: str
    product_evidence: FrozenEvidenceShape
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    structural_score: float
    fused_score: float


class RetrievalContractAdapterTest(unittest.TestCase):
    def test_adapter_constructs_exact_frozen_candidate_and_evidence_shapes(self) -> None:
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
            plan = SimpleNamespace(
                text_query="waterproof hiking shoes",
                must_terms=(),
                should_terms=("walking",),
                exclude_terms=(),
                filters={"budget": {"max": 80}},
                attribute_values={"category": ("shoes",), "budget": (80,)},
                mode="buying",
                intent_version=1,
                revalidation_flags=frozenset(),
                no_preference=frozenset(),
                profile_bias=SimpleNamespace(terms=(), weight=0.0),
                route_policy="hybrid",
                embedding_route_id=manifest.embedding_route_id,
                index_id=manifest.index_id,
            )
            with HybridRetriever(
                catalog,
                dense_index=index,
                query_embedder=embedder,
            ) as retriever:
                adapter = ContractRetrieverAdapter(
                    retriever,
                    candidate_factory=FrozenCandidateShape,
                    evidence_factory=FrozenEvidenceShape,
                )
                candidates = adapter.search(plan, 5)

        self.assertTrue(candidates)
        self.assertTrue(all(isinstance(item, FrozenCandidateShape) for item in candidates))
        self.assertTrue(all(item.parent_asin in catalog for item in candidates))
        evidence = candidates[0].product_evidence
        self.assertEqual(set(evidence.constraint_outcomes), set(evidence.attribute_values))
        self.assertIn(evidence.constraint_outcomes["budget"], {"match", "unknown"})
        self.assertEqual(evidence.route_details["dense_index_id"], manifest.index_id)


if __name__ == "__main__":
    unittest.main()
