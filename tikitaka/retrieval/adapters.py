"""Mechanical conversion from retrieval evidence to canonical shared contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from tikitaka.contracts import (
    Attribute,
    Candidate,
    EvidenceOutcome,
    ProductEvidence,
    Retriever,
    SearchPlan,
)

from .structured import ATTRIBUTE_NAMES

if TYPE_CHECKING:
    from .hybrid import HybridRetrievalHit
    from .retriever import RetrievalHit
    ContractSourceHit = HybridRetrievalHit | RetrievalHit
else:
    ContractSourceHit = Any


def _constraint_outcomes(hit: ContractSourceHit) -> MappingProxyType:
    priorities = {
        EvidenceOutcome.UNKNOWN: 0,
        EvidenceOutcome.MATCH: 1,
        EvidenceOutcome.CONTRADICTION: 2,
    }
    outcomes = {Attribute(attribute): EvidenceOutcome.UNKNOWN for attribute in ATTRIBUTE_NAMES}
    for evaluation in hit.constraint_evaluations:
        attribute = Attribute(evaluation.attribute)
        outcome = EvidenceOutcome(evaluation.outcome)
        if priorities[outcome] > priorities[outcomes[attribute]]:
            outcomes[attribute] = outcome
    return MappingProxyType(outcomes)


def contract_product_evidence(hit: ContractSourceHit) -> ProductEvidence:
    """Construct Person 4's canonical ProductEvidence contract."""

    attribute_values = MappingProxyType(
        {
            Attribute(attribute): hit.structured_evidence.for_attribute(attribute).values
            for attribute in ATTRIBUTE_NAMES
        }
    )
    reliability = MappingProxyType(
        {
            Attribute(attribute): hit.structured_evidence.for_attribute(attribute).reliability
            for attribute in ATTRIBUTE_NAMES
        }
    )
    return ProductEvidence(
        matched_fields=hit.matched_fields,
        supporting_snippets=hit.supporting_snippets,
        constraint_outcomes=_constraint_outcomes(hit),
        attribute_values=attribute_values,
        evidence_reliability=reliability,
        unknown_fields=hit.structured_evidence.unknown_fields,
        route_details=hit.route_details,
        profile_contribution=hit.profile_contribution,
    )


def contract_candidate(hit: ContractSourceHit) -> Candidate:
    """Construct Person 4's canonical Candidate contract."""

    return Candidate(
        parent_asin=hit.parent_asin,
        product_evidence=contract_product_evidence(hit),
        sparse_rank=hit.sparse_rank,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        dense_score=hit.dense_score,
        structural_score=hit.structural_score,
        fused_score=hit.fused_score,
    )


class ContractRetrieverAdapter:
    """Compatibility wrapper; HybridRetriever now satisfies Retriever directly."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]:
        return self.retriever.search(plan, limit)
