"""Deterministic M1 sparse plus structured retrieval pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from tikitaka.contracts import Candidate

from .adapters import contract_candidate
from .catalog import ProductCatalog, ProductDocument
from .request import RetrievalConstraint, RetrievalRequest, request_from_search_plan
from .sparse import SparseHit, SparseIndex, SparseIndexConfig
from .structured import (
    ConstraintEvaluation,
    StructuredProductEvidence,
    evaluate_constraint,
    extract_structured_evidence,
)
from .text import build_sparse_fields, query_terms


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    pool_depth: int = 200
    rrf_k: int = 60
    structural_weight: float = 0.05
    hard_filter_reliability: float = 0.8
    hard_match_weight: float = 1.0
    soft_match_weight: float = 0.35
    hard_contradiction_weight: float = -1.0
    soft_contradiction_weight: float = -0.25
    exclude_term_penalty: float = -0.2
    # Profile data is an optional session-local prior for otherwise vague
    # requests. Once dialogue constraints exist, its contribution is disabled.
    max_profile_contribution: float = 0.005

    def __post_init__(self) -> None:
        if self.pool_depth <= 0 or self.rrf_k <= 0:
            raise ValueError("pool_depth and rrf_k must be positive")
        numeric_values = (
            self.structural_weight,
            self.hard_filter_reliability,
            self.hard_match_weight,
            self.soft_match_weight,
            self.hard_contradiction_weight,
            self.soft_contradiction_weight,
            self.exclude_term_penalty,
            self.max_profile_contribution,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("retrieval weights must be finite")
        if not 0.0 <= self.hard_filter_reliability <= 1.0:
            raise ValueError("hard_filter_reliability must be within [0.0, 1.0]")
        if min(
            self.structural_weight,
            self.hard_match_weight,
            self.soft_match_weight,
            self.max_profile_contribution,
        ) < 0:
            raise ValueError("retrieval boost weights must be non-negative")
        if max(
            self.hard_contradiction_weight,
            self.soft_contradiction_weight,
            self.exclude_term_penalty,
        ) > 0:
            raise ValueError("retrieval contradiction weights must be non-positive")


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    parent_asin: str
    product: ProductDocument
    structured_evidence: StructuredProductEvidence
    constraint_evaluations: tuple[ConstraintEvaluation, ...]
    sparse_rank: int
    sparse_score: float
    dense_rank: int | None
    dense_score: float | None
    structural_score: float
    fused_score: float
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]
    matched_exclude_terms: tuple[str, ...]
    route_details: Mapping[str, object]
    profile_contribution: float


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    route: str
    sparse_candidates: int
    hard_filtered_candidates: int
    returned_candidates: int
    intent_version: int
    elapsed_ms: float
    sparse_manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    hits: tuple[RetrievalHit, ...]
    diagnostics: RetrievalDiagnostics


def _constraint_score(
    constraint: RetrievalConstraint,
    evaluation: ConstraintEvaluation,
    config: RetrievalConfig,
) -> float:
    if evaluation.outcome == "match":
        weight = config.hard_match_weight if constraint.strength == "hard" else config.soft_match_weight
        return weight * evaluation.reliability
    if evaluation.outcome == "contradiction":
        weight = (
            config.hard_contradiction_weight
            if constraint.strength == "hard"
            else config.soft_contradiction_weight
        )
        return weight * evaluation.reliability
    return 0.0


def _matched_terms(product: ProductDocument, terms: tuple[str, ...]) -> tuple[str, ...]:
    if not terms:
        return ()
    fields = build_sparse_fields(product)
    product_terms = set(query_terms(*fields.ordered_values(), max_terms=20_000))
    normalized = query_terms(*terms, max_terms=max(1, len(terms) * 5))
    return tuple(term for term in normalized if term in product_terms)


def _positive_constraint_terms(
    constraints: tuple[RetrievalConstraint, ...],
) -> tuple[str, ...]:
    """Expose active positive intent to lexical recall without encoding filters."""

    values: list[str] = []
    for constraint in constraints:
        if constraint.polarity == "include" and constraint.attribute != "budget":
            values.extend(str(value) for value in constraint.values)
    return tuple(values)


class SparseStructuredRetriever:
    """M1 retriever that remains deterministic and contains no session state."""

    def __init__(
        self,
        catalog: ProductCatalog,
        *,
        sparse_config: SparseIndexConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.sparse = SparseIndex(catalog, config=sparse_config)
        self.config = retrieval_config or RetrievalConfig()
        self._evidence_cache: dict[str, StructuredProductEvidence] = {}

    def _evidence(self, product: ProductDocument) -> StructuredProductEvidence:
        cached = self._evidence_cache.get(product.parent_asin)
        if cached is None:
            cached = extract_structured_evidence(product)
            self._evidence_cache[product.parent_asin] = cached
        return cached

    def _evaluate(
        self,
        evidence: StructuredProductEvidence,
        constraints: tuple[RetrievalConstraint, ...],
    ) -> tuple[tuple[ConstraintEvaluation, ...], bool, float]:
        evaluations: list[ConstraintEvaluation] = []
        structural_score = 0.0
        hard_conflict = False
        for constraint in constraints:
            evaluation = evaluate_constraint(
                evidence,
                attribute=constraint.attribute,
                desired_values=constraint.values,
                polarity=constraint.polarity,
                operator=constraint.operator,
            )
            evaluations.append(evaluation)
            structural_score += _constraint_score(constraint, evaluation, self.config)
            if (
                constraint.strength == "hard"
                and not constraint.needs_revalidation
                and evaluation.outcome == "contradiction"
                and evaluation.reliability >= self.config.hard_filter_reliability
            ):
                hard_conflict = True
        return tuple(evaluations), hard_conflict, structural_score

    def retrieve(self, plan_or_request: object, *, limit: int = 100) -> RetrievalResult:
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        request = (
            plan_or_request
            if isinstance(plan_or_request, RetrievalRequest)
            else request_from_search_plan(plan_or_request)
        )
        started = time.perf_counter()
        pool_limit = max(limit, self.config.pool_depth)
        sparse_hits = self.sparse.search(
            request.text_query,
            must_terms=request.must_terms,
            should_terms=(*request.should_terms, *_positive_constraint_terms(request.constraints)),
            limit=pool_limit,
        )
        hits: list[RetrievalHit] = []
        filtered = 0
        for sparse_hit in sparse_hits:
            product = self.catalog.require(sparse_hit.parent_asin)
            evidence = self._evidence(product)
            evaluations, hard_conflict, structural_score = self._evaluate(
                evidence, request.constraints
            )
            if hard_conflict:
                filtered += 1
                continue
            matched_excludes = _matched_terms(product, request.exclude_terms)
            structural_score += self.config.exclude_term_penalty * len(matched_excludes)
            if request.constraints:
                profile_contribution = 0.0
            else:
                profile_matches = _matched_terms(product, request.profile_terms)
                profile_term_count = len(query_terms(*request.profile_terms))
                profile_ratio = len(profile_matches) / max(1, profile_term_count)
                profile_contribution = min(
                    self.config.max_profile_contribution,
                    request.profile_weight * self.config.max_profile_contribution * profile_ratio,
                )
            fused_score = (
                1.0 / (self.config.rrf_k + sparse_hit.rank)
                + self.config.structural_weight * structural_score
                + profile_contribution
            )
            constraint_snippets = tuple(
                snippet
                for evaluation in evaluations
                for snippet in evaluation.snippets
            )
            snippets = tuple(dict.fromkeys((*sparse_hit.supporting_snippets, *constraint_snippets)))[:6]
            hits.append(
                RetrievalHit(
                    parent_asin=product.parent_asin,
                    product=product,
                    structured_evidence=evidence,
                    constraint_evaluations=evaluations,
                    sparse_rank=sparse_hit.rank,
                    sparse_score=sparse_hit.score,
                    dense_rank=None,
                    dense_score=None,
                    structural_score=structural_score,
                    fused_score=fused_score,
                    matched_fields=sparse_hit.matched_fields,
                    supporting_snippets=snippets,
                    matched_exclude_terms=matched_excludes,
                    route_details=MappingProxyType(
                        {
                            "sparse_rank": sparse_hit.rank,
                            "sparse_score": sparse_hit.score,
                            "dense_rank": None,
                            "dense_score": None,
                            "sparse_index_id": self.sparse.manifest.engine,
                            "dense_index_id": None,
                            "embedding_route_id": None,
                        }
                    ),
                    profile_contribution=profile_contribution,
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.fused_score,
                -hit.structural_score,
                hit.sparse_rank,
                hit.parent_asin,
            )
        )
        unique_hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.parent_asin not in seen:
                seen.add(hit.parent_asin)
                unique_hits.append(hit)
                if len(unique_hits) >= limit:
                    break
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        diagnostics = RetrievalDiagnostics(
            route="sparse_structured",
            sparse_candidates=len(sparse_hits),
            hard_filtered_candidates=filtered,
            returned_candidates=len(unique_hits),
            intent_version=request.intent_version,
            elapsed_ms=elapsed_ms,
            sparse_manifest=MappingProxyType(self.sparse.manifest.as_dict()),
        )
        return RetrievalResult(hits=tuple(unique_hits), diagnostics=diagnostics)

    def search(self, plan_or_request: object, limit: int) -> list[Candidate]:
        """Satisfy the canonical Retriever protocol on the no-network route."""

        return [
            contract_candidate(hit)
            for hit in self.retrieve(plan_or_request, limit=limit).hits
        ]

    def search_hits(self, plan_or_request: object, limit: int) -> list[RetrievalHit]:
        return list(self.retrieve(plan_or_request, limit=limit).hits)

    def close(self) -> None:
        self.sparse.close()

    def __enter__(self) -> "SparseStructuredRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
