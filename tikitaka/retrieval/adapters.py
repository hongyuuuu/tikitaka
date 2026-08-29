"""Mechanical adapters into Person 4-owned frozen contract classes."""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable

from .hybrid import HybridRetrievalHit, HybridRetriever
from .structured import ATTRIBUTE_NAMES


def _constraint_outcomes(hit: HybridRetrievalHit) -> MappingProxyType:
    priorities = {"unknown": 0, "match": 1, "contradiction": 2}
    outcomes = {attribute: "unknown" for attribute in ATTRIBUTE_NAMES}
    for evaluation in hit.constraint_evaluations:
        current = outcomes[evaluation.attribute]
        if priorities[evaluation.outcome] > priorities[current]:
            outcomes[evaluation.attribute] = evaluation.outcome
    return MappingProxyType(outcomes)


def contract_product_evidence(
    hit: HybridRetrievalHit,
    evidence_factory: Callable[..., object],
) -> object:
    """Construct frozen ProductEvidence without redefining the shared class."""

    attribute_values = MappingProxyType(
        {
            attribute: hit.structured_evidence.for_attribute(attribute).values
            for attribute in ATTRIBUTE_NAMES
        }
    )
    reliability = MappingProxyType(
        {
            attribute: hit.structured_evidence.for_attribute(attribute).reliability
            for attribute in ATTRIBUTE_NAMES
        }
    )
    return evidence_factory(
        matched_fields=hit.matched_fields,
        supporting_snippets=hit.supporting_snippets,
        constraint_outcomes=_constraint_outcomes(hit),
        attribute_values=attribute_values,
        evidence_reliability=reliability,
        unknown_fields=hit.structured_evidence.unknown_fields,
        route_details=hit.route_details,
        profile_contribution=hit.profile_contribution,
    )


def contract_candidate(
    hit: HybridRetrievalHit,
    *,
    candidate_factory: Callable[..., object],
    evidence_factory: Callable[..., object],
) -> object:
    """Construct frozen Candidate using only its accepted 0.1.0 keyword shape."""

    return candidate_factory(
        parent_asin=hit.parent_asin,
        product_evidence=contract_product_evidence(hit, evidence_factory),
        sparse_rank=hit.sparse_rank,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        dense_score=hit.dense_score,
        structural_score=hit.structural_score,
        fused_score=hit.fused_score,
    )


class ContractRetrieverAdapter:
    """Satisfy the frozen Retriever protocol once canonical classes are supplied."""

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        candidate_factory: Callable[..., object],
        evidence_factory: Callable[..., object],
    ) -> None:
        self.retriever = retriever
        self.candidate_factory = candidate_factory
        self.evidence_factory = evidence_factory

    def search(self, plan: object, limit: int) -> list[object]:
        result = self.retriever.retrieve(plan, limit=limit)
        return [
            contract_candidate(
                hit,
                candidate_factory=self.candidate_factory,
                evidence_factory=self.evidence_factory,
            )
            for hit in result.hits
        ]
