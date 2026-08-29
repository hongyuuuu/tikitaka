"""Stateless sparse+dense+structured retrieval with safe route degradation."""

from __future__ import annotations

import time
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from tikitaka.contracts import Candidate

from .adapters import contract_candidate
from .catalog import ProductCatalog, ProductDocument
from .dense import (
    DenseArtifactError,
    DenseIndex,
    DenseRouteError,
    assert_embedder_matches_manifest,
    embed_query_for_index,
)
from .fusion import RRFConfig, reciprocal_rank_fusion, route_overlap
from .manifests import assert_dense_manifest_compatible
from .request import RetrievalConstraint, RetrievalRequest, request_from_search_plan
from .retriever import RetrievalConfig, _constraint_score, _matched_terms
from .sparse import SparseIndex, SparseIndexConfig
from .structured import (
    ATTRIBUTE_NAMES,
    ConstraintEvaluation,
    StructuredProductEvidence,
    evaluate_constraint,
    extract_structured_evidence,
)
from .text import build_dense_query
from .text import query_terms


@dataclass(frozen=True, slots=True)
class HybridConfig:
    sparse_depth: int = 200
    dense_depth: int = 200
    fused_depth: int = 200
    rrf_k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    buying_sparse_multiplier: float = 1.1
    buying_dense_multiplier: float = 0.9
    browsing_sparse_multiplier: float = 0.9
    browsing_dense_multiplier: float = 1.1
    ranking: RetrievalConfig = field(default_factory=RetrievalConfig)

    def __post_init__(self) -> None:
        if min(self.sparse_depth, self.dense_depth, self.fused_depth, self.rrf_k) <= 0:
            raise ValueError("hybrid route depths and RRF k must be positive")
        weights = (
            self.sparse_weight,
            self.dense_weight,
            self.buying_sparse_multiplier,
            self.buying_dense_multiplier,
            self.browsing_sparse_multiplier,
            self.browsing_dense_multiplier,
        )
        if not all(math.isfinite(value) for value in weights):
            raise ValueError("hybrid route weights and multipliers must be finite")
        if any(value < 0 for value in weights):
            raise ValueError("hybrid route weights and multipliers must be non-negative")
        if self.sparse_weight == 0 and self.dense_weight == 0:
            raise ValueError("at least one hybrid route weight must be positive")
        if self.buying_sparse_multiplier == 0 and self.buying_dense_multiplier == 0:
            raise ValueError("buying mode must retain at least one route")
        if self.browsing_sparse_multiplier == 0 and self.browsing_dense_multiplier == 0:
            raise ValueError("browsing mode must retain at least one route")

    def weights_for_mode(self, mode: str) -> tuple[float, float]:
        if mode == "buying":
            return (
                self.sparse_weight * self.buying_sparse_multiplier,
                self.dense_weight * self.buying_dense_multiplier,
            )
        if mode == "browsing":
            return (
                self.sparse_weight * self.browsing_sparse_multiplier,
                self.dense_weight * self.browsing_dense_multiplier,
            )
        return self.sparse_weight, self.dense_weight


@dataclass(frozen=True, slots=True)
class HybridRetrievalHit:
    parent_asin: str
    product: ProductDocument
    structured_evidence: StructuredProductEvidence
    constraint_evaluations: tuple[ConstraintEvaluation, ...]
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    route_score: float
    structural_score: float
    fused_score: float
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]
    matched_exclude_terms: tuple[str, ...]
    route_details: Mapping[str, object]
    profile_contribution: float

    @property
    def best_route_rank(self) -> int:
        ranks = tuple(rank for rank in (self.sparse_rank, self.dense_rank) if rank is not None)
        return min(ranks) if ranks else 2**31 - 1


@dataclass(frozen=True, slots=True)
class HybridDiagnostics:
    requested_route: str
    executed_route: str
    route_failures: tuple[str, ...]
    sparse_candidates: int
    dense_candidates: int
    fused_candidates: int
    hard_filtered_candidates: int
    returned_candidates: int
    route_overlap: Mapping[int, int]
    top_score_margin: float
    top_score_concentration: float
    effective_candidate_count: float
    attribute_distributions: Mapping[str, tuple[tuple[str, int], ...]]
    missing_attribute_rates: Mapping[str, float]
    constraint_outcome_counts: Mapping[str, Mapping[str, int]]
    route_timings_ms: Mapping[str, float]
    manifest_ids: Mapping[str, str]
    intent_version: int


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    hits: tuple[HybridRetrievalHit, ...]
    diagnostics: HybridDiagnostics


def _positive_constraint_terms(constraints: tuple[RetrievalConstraint, ...]) -> tuple[str, ...]:
    return tuple(
        str(value)
        for constraint in constraints
        if constraint.polarity == "include" and constraint.attribute != "budget"
        for value in constraint.values
    )


def _score_distribution(hits: list[HybridRetrievalHit]) -> tuple[float, float, float]:
    if not hits:
        return 0.0, 0.0, 0.0
    margin = hits[0].fused_score - hits[1].fused_score if len(hits) > 1 else 0.0
    values = [hit.fused_score for hit in hits[:10]]
    minimum = min(values)
    shifted = [value - minimum + 1e-12 for value in values]
    total = sum(shifted)
    probabilities = [value / total for value in shifted]
    concentration = max(probabilities)
    effective_count = 1.0 / sum(probability * probability for probability in probabilities)
    return margin, concentration, effective_count


def _candidate_diagnostics(
    hits: list[HybridRetrievalHit],
) -> tuple[
    Mapping[str, tuple[tuple[str, int], ...]],
    Mapping[str, float],
    Mapping[str, Mapping[str, int]],
]:
    competitive = hits[:100]
    distributions: dict[str, tuple[tuple[str, int], ...]] = {}
    missing_rates: dict[str, float] = {}
    outcome_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for attribute in ATTRIBUTE_NAMES:
        values: Counter[str] = Counter()
        missing = 0
        for hit in competitive:
            evidence = hit.structured_evidence.for_attribute(attribute)
            if evidence.known:
                values.update(str(value) for value in evidence.values)
            else:
                missing += 1
        distributions[attribute] = tuple(
            sorted(values.items(), key=lambda item: (-item[1], item[0]))[:20]
        )
        missing_rates[attribute] = missing / len(competitive) if competitive else 1.0
    for hit in competitive:
        for evaluation in hit.constraint_evaluations:
            outcome_counts[evaluation.attribute][evaluation.outcome] += 1
    frozen_outcomes = {
        attribute: MappingProxyType(dict(sorted(counts.items())))
        for attribute, counts in sorted(outcome_counts.items())
    }
    return (
        MappingProxyType(distributions),
        MappingProxyType(missing_rates),
        MappingProxyType(frozen_outcomes),
    )


class HybridRetriever:
    """Full Role 2 retriever; cached state is catalog-derived and session-agnostic."""

    def __init__(
        self,
        catalog: ProductCatalog,
        *,
        dense_index: DenseIndex | None = None,
        query_embedder: object | None = None,
        sparse_config: SparseIndexConfig | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        if (dense_index is None) != (query_embedder is None):
            raise ValueError("dense_index and query_embedder must be supplied together")
        if dense_index is not None:
            try:
                assert_dense_manifest_compatible(dense_index.manifest, catalog)
                assert_embedder_matches_manifest(query_embedder, dense_index.manifest)
            except (ValueError, DenseRouteError) as error:
                raise ValueError(
                    "dense index/query embedder does not match hybrid configuration"
                ) from error
        self.catalog = catalog
        self.sparse = SparseIndex(catalog, config=sparse_config)
        self.dense = dense_index
        self.query_embedder = query_embedder
        self.config = config or HybridConfig()
        self._evidence_cache: dict[str, StructuredProductEvidence] = {}

    def _evidence(self, product: ProductDocument) -> StructuredProductEvidence:
        evidence = self._evidence_cache.get(product.parent_asin)
        if evidence is None:
            evidence = extract_structured_evidence(product)
            self._evidence_cache[product.parent_asin] = evidence
        return evidence

    def _dense_pin_failure(self, request: RetrievalRequest) -> str | None:
        if self.dense is None or self.query_embedder is None:
            return "dense_route_unavailable"
        manifest = self.dense.manifest
        if request.embedding_route_id is not None:
            if request.embedding_route_id != manifest.route_id:
                return "embedding_route_mismatch"
            if request.index_id != manifest.index_id:
                return "dense_index_mismatch"
        return None

    def _evaluate_constraints(
        self,
        evidence: StructuredProductEvidence,
        constraints: tuple[RetrievalConstraint, ...],
    ) -> tuple[tuple[ConstraintEvaluation, ...], bool, float]:
        evaluations: list[ConstraintEvaluation] = []
        hard_conflict = False
        structural_score = 0.0
        for constraint in constraints:
            evaluation = evaluate_constraint(
                evidence,
                attribute=constraint.attribute,
                desired_values=constraint.values,
                polarity=constraint.polarity,
                operator=constraint.operator,
            )
            evaluations.append(evaluation)
            structural_score += _constraint_score(
                constraint,
                evaluation,
                self.config.ranking,
            )
            if (
                self.config.ranking.hard_filtering
                and constraint.strength == "hard"
                and not constraint.needs_revalidation
                and evaluation.outcome == "contradiction"
                and evaluation.reliability >= self.config.ranking.hard_filter_reliability
            ):
                hard_conflict = True
        return tuple(evaluations), hard_conflict, structural_score

    def retrieve(self, plan_or_request: object, *, limit: int = 100) -> HybridRetrievalResult:
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        request = (
            plan_or_request
            if isinstance(plan_or_request, RetrievalRequest)
            else request_from_search_plan(plan_or_request)
        )
        started = time.perf_counter()
        timings: dict[str, float] = {}
        failures: list[str] = []
        requested = request.route_policy
        run_sparse = requested in {"auto", "sparse", "hybrid"}
        run_dense = requested in {"auto", "dense", "hybrid"}
        pin_failure = self._dense_pin_failure(request) if run_dense else None
        if pin_failure is not None:
            failures.append(pin_failure)
            run_dense = False
            run_sparse = True

        sparse_hits: list[object] = []
        if run_sparse:
            route_started = time.perf_counter()
            sparse_hits = self.sparse.search(
                request.text_query,
                must_terms=request.must_terms,
                should_terms=(
                    *request.should_terms,
                    *_positive_constraint_terms(request.constraints),
                ),
                limit=max(limit, self.config.sparse_depth),
            )
            timings["sparse"] = (time.perf_counter() - route_started) * 1_000.0

        dense_hits: list[object] = []
        if run_dense and self.dense is not None and self.query_embedder is not None:
            route_started = time.perf_counter()
            dense_query = build_dense_query(
                request.text_query,
                must_terms=request.must_terms,
                should_terms=request.should_terms,
                constraints=request.constraints,
            )
            try:
                query_vector = embed_query_for_index(
                    self.query_embedder,
                    self.dense,
                    dense_query,
                )
                dense_hits = self.dense.search(
                    query_vector,
                    limit=max(limit, self.config.dense_depth),
                )
            except (DenseRouteError, DenseArtifactError):
                failures.append("dense_query_failed")
                run_dense = False
                if not run_sparse:
                    fallback_started = time.perf_counter()
                    sparse_hits = self.sparse.search(
                        request.text_query,
                        must_terms=request.must_terms,
                        should_terms=(
                            *request.should_terms,
                            *_positive_constraint_terms(request.constraints),
                        ),
                        limit=max(limit, self.config.sparse_depth),
                    )
                    timings["sparse"] = (
                        time.perf_counter() - fallback_started
                    ) * 1_000.0
                    run_sparse = True
            timings["dense"] = (time.perf_counter() - route_started) * 1_000.0

        sparse_weight, dense_weight = self.config.weights_for_mode(request.mode)
        if not sparse_hits:
            sparse_weight = 0.0
        if not dense_hits:
            dense_weight = 0.0
        if sparse_weight == 0.0 and dense_weight == 0.0:
            fused_route_hits = []
        else:
            fused_route_hits = reciprocal_rank_fusion(
                sparse_hits,
                dense_hits,
                config=RRFConfig(
                    k=self.config.rrf_k,
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    candidate_limit=max(
                        limit,
                        self.config.fused_depth,
                        len(sparse_hits) + len(dense_hits),
                    ),
                ),
                valid_ids=self.catalog.ids,
            )

        sparse_by_id = {str(hit.parent_asin): hit for hit in sparse_hits}
        ranked: list[HybridRetrievalHit] = []
        hard_filtered = 0
        for route_hit in fused_route_hits:
            product = self.catalog.require(route_hit.parent_asin)
            evidence = self._evidence(product)
            evaluations, hard_conflict, structural_score = self._evaluate_constraints(
                evidence,
                request.constraints,
            )
            if hard_conflict:
                hard_filtered += 1
                continue
            matched_excludes = _matched_terms(product, request.exclude_terms)
            structural_score += (
                self.config.ranking.exclude_term_penalty * len(matched_excludes)
            )
            if request.constraints:
                profile_contribution = 0.0
            else:
                profile_matches = _matched_terms(product, request.profile_terms)
                profile_term_count = len(query_terms(*request.profile_terms))
                profile_ratio = len(profile_matches) / max(1, profile_term_count)
                profile_contribution = min(
                    self.config.ranking.max_profile_contribution,
                    request.profile_weight
                    * self.config.ranking.max_profile_contribution
                    * profile_ratio,
                )
            fused_score = (
                route_hit.route_score
                + self.config.ranking.structural_weight * structural_score
                + profile_contribution
            )
            sparse_hit = sparse_by_id.get(product.parent_asin)
            matched_fields = (
                tuple(sparse_hit.matched_fields) if sparse_hit is not None else ("dense_text",)
            )
            route_snippets = (
                tuple(sparse_hit.supporting_snippets)
                if sparse_hit is not None
                else tuple(
                    snippet
                    for snippet in (
                        f"title: {product.title[:240]}" if product.title else "",
                        "categories: " + " > ".join(product.categories[-4:]),
                    )
                    if snippet
                )
            )
            constraint_snippets = tuple(
                snippet for evaluation in evaluations for snippet in evaluation.snippets
            )
            snippets = tuple(dict.fromkeys((*route_snippets, *constraint_snippets)))[:6]
            dense_manifest = None if self.dense is None else self.dense.manifest
            route_details = MappingProxyType(
                {
                    "sparse_rank": route_hit.sparse_rank,
                    "sparse_score": route_hit.sparse_score,
                    "dense_rank": route_hit.dense_rank,
                    "dense_score": route_hit.dense_score,
                    "route_score": route_hit.route_score,
                    "sparse_index_id": self.sparse.manifest.engine,
                    "dense_index_id": None if dense_manifest is None else dense_manifest.index_id,
                    "dense_backend": None if self.dense is None else self.dense.backend,
                    "embedding_route_id": (
                        None if dense_manifest is None else dense_manifest.route_id
                    ),
                }
            )
            ranked.append(
                HybridRetrievalHit(
                    parent_asin=product.parent_asin,
                    product=product,
                    structured_evidence=evidence,
                    constraint_evaluations=evaluations,
                    sparse_rank=route_hit.sparse_rank,
                    sparse_score=route_hit.sparse_score,
                    dense_rank=route_hit.dense_rank,
                    dense_score=route_hit.dense_score,
                    route_score=route_hit.route_score,
                    structural_score=structural_score,
                    fused_score=fused_score,
                    matched_fields=matched_fields,
                    supporting_snippets=snippets,
                    matched_exclude_terms=matched_excludes,
                    route_details=route_details,
                    profile_contribution=profile_contribution,
                )
            )
        ranked.sort(
            key=lambda hit: (
                -hit.fused_score,
                -hit.structural_score,
                hit.best_route_rank,
                hit.sparse_rank if hit.sparse_rank is not None else 2**31 - 1,
                hit.dense_rank if hit.dense_rank is not None else 2**31 - 1,
                hit.parent_asin,
            )
        )
        returned = ranked[:limit]
        margin, concentration, effective_count = _score_distribution(ranked)
        distributions, missing_rates, outcome_counts = _candidate_diagnostics(ranked)
        executed = (
            "hybrid"
            if sparse_hits and dense_hits
            else "dense"
            if dense_hits
            else "sparse_fallback"
            if failures
            else "sparse"
        )
        dense_manifest = None if self.dense is None else self.dense.manifest
        timings["total"] = (time.perf_counter() - started) * 1_000.0
        diagnostics = HybridDiagnostics(
            requested_route=requested,
            executed_route=executed,
            route_failures=tuple(dict.fromkeys(failures)),
            sparse_candidates=len(sparse_hits),
            dense_candidates=len(dense_hits),
            fused_candidates=len(fused_route_hits),
            hard_filtered_candidates=hard_filtered,
            returned_candidates=len(returned),
            route_overlap=MappingProxyType(route_overlap(sparse_hits, dense_hits)),
            top_score_margin=margin,
            top_score_concentration=concentration,
            effective_candidate_count=effective_count,
            attribute_distributions=distributions,
            missing_attribute_rates=missing_rates,
            constraint_outcome_counts=outcome_counts,
            route_timings_ms=MappingProxyType(
                {key: round(value, 6) for key, value in sorted(timings.items())}
            ),
            manifest_ids=MappingProxyType(
                {
                    "sparse": self.sparse.manifest.engine,
                    "dense": "" if dense_manifest is None else dense_manifest.index_id,
                    "dense_backend": "" if self.dense is None else self.dense.backend,
                }
            ),
            intent_version=request.intent_version,
        )
        return HybridRetrievalResult(tuple(returned), diagnostics)

    def search(self, plan_or_request: object, limit: int) -> list[Candidate]:
        """Satisfy Person 4's canonical Retriever protocol directly."""

        return [
            contract_candidate(hit)
            for hit in self.retrieve(plan_or_request, limit=limit).hits
        ]

    def search_hits(self, plan_or_request: object, limit: int) -> list[HybridRetrievalHit]:
        """Return internal evidence-rich hits for Role 2 diagnostics and tests."""

        return list(self.retrieve(plan_or_request, limit=limit).hits)

    def close(self) -> None:
        self.sparse.close()

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
