"""Deterministic candidate-pool uncertainty diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from tikitaka.ranking.constraints import (
    active_constraints,
    attribute_values,
    enum_value,
    mapping_lookup,
    normalized_value,
    unique_candidates,
)
from tikitaka.ranking.deterministic import normalized_score_magnitudes


ALLOWED_ATTRIBUTES: tuple[str, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)


@dataclass(frozen=True)
class DiagnosticsConfig:
    competitive_limit: int = 50
    route_top_k: int = 20
    top_k_boundary: int = 10
    score_temperature: float = 0.18

    def __post_init__(self) -> None:
        if min(self.competitive_limit, self.route_top_k, self.top_k_boundary) <= 0:
            raise ValueError("diagnostic limits must be positive")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be positive")


@dataclass(frozen=True)
class CandidatePoolDiagnostics:
    candidate_count: int
    effective_candidate_mass: float
    score_concentration: float
    lead_margin: float
    top_k_boundary_margin: float
    route_disagreement: float
    constraint_coverage: float
    metadata_sufficiency: float
    attribute_uncertainty: Mapping[str, float]
    attribute_coverage: Mapping[str, float]
    attribute_distributions: Mapping[str, Mapping[str, float]]
    top_k_instability: float


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _normalized_scores(candidates: Sequence[object]) -> tuple[float, ...]:
    values = tuple(_finite(getattr(item, "fused_score", 0.0)) for item in candidates)
    return normalized_score_magnitudes(values)


def _softmax(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not values:
        return ()
    high = max(values)
    exponents = tuple(math.exp((value - high) / temperature) for value in values)
    total = sum(exponents)
    return tuple(value / total for value in exponents)


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    nonzero = [value for value in probabilities if value > 0]
    if len(nonzero) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in nonzero)
    return min(1.0, max(0.0, entropy / math.log(len(nonzero))))


def _margin(values: Sequence[float], left: int, right: int) -> float:
    if len(values) <= right:
        return 1.0
    return min(1.0, max(0.0, values[left] - values[right]))


def _route_ids(candidates: Sequence[object], rank_field: str, limit: int) -> tuple[str, ...]:
    ranked = [item for item in candidates if getattr(item, rank_field, None) is not None]
    ranked.sort(
        key=lambda item: (
            int(getattr(item, rank_field)),
            str(getattr(item, "parent_asin")),
        )
    )
    return tuple(str(getattr(item, "parent_asin")) for item in ranked[:limit])


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _route_disagreement(candidates: Sequence[object], limit: int) -> float:
    sparse = _route_ids(candidates, "sparse_rank", limit)
    dense = _route_ids(candidates, "dense_rank", limit)
    if not sparse or not dense:
        return 0.0
    return 1.0 - _jaccard(sparse, dense)


def _constraint_coverage(state: object, candidates: Sequence[object]) -> float:
    constraints = active_constraints(state)
    if not constraints:
        return 0.0
    covered = 0.0
    total = len(constraints) * max(1, len(candidates))
    for candidate in candidates:
        outcomes = getattr(getattr(candidate, "product_evidence", None), "constraint_outcomes", {}) or {}
        for constraint in constraints:
            outcome = enum_value(mapping_lookup(outcomes, getattr(constraint, "attribute", ""), "unknown"))
            covered += outcome in {"match", "contradiction"}
    return covered / total


def _attribute_statistics(
    candidates: Sequence[object],
) -> tuple[dict[str, float], dict[str, float], dict[str, Mapping[str, float]]]:
    uncertainty: dict[str, float] = {}
    coverage: dict[str, float] = {}
    distributions: dict[str, Mapping[str, float]] = {}
    count = max(1, len(candidates))
    for attribute in ALLOWED_ATTRIBUTES:
        values_counter: Counter[str] = Counter()
        known = 0
        for candidate in candidates:
            values = mapping_lookup(attribute_values(candidate), attribute, ()) or ()
            normalized = {normalized_value(value) for value in values}
            if normalized:
                known += 1
                share = 1.0 / len(normalized)
                for value in normalized:
                    values_counter[value] += share
        coverage[attribute] = known / count
        total = sum(values_counter.values())
        distribution = {
            value: amount / total for value, amount in sorted(values_counter.items())
        } if total else {}
        distributions[attribute] = distribution
        uncertainty[attribute] = _normalized_entropy(tuple(distribution.values()))
    return uncertainty, coverage, distributions


def diagnose_pool(
    state: object,
    candidates: Sequence[object],
    config: DiagnosticsConfig | None = None,
) -> CandidatePoolDiagnostics:
    policy = config or DiagnosticsConfig()
    competitive = unique_candidates(candidates)[: policy.competitive_limit]
    if not competitive:
        empty = {attribute: 0.0 for attribute in ALLOWED_ATTRIBUTES}
        distributions = {attribute: {} for attribute in ALLOWED_ATTRIBUTES}
        return CandidatePoolDiagnostics(
            candidate_count=0,
            effective_candidate_mass=0.0,
            score_concentration=0.0,
            lead_margin=0.0,
            top_k_boundary_margin=0.0,
            route_disagreement=0.0,
            constraint_coverage=0.0,
            metadata_sufficiency=0.0,
            attribute_uncertainty=empty,
            attribute_coverage=empty.copy(),
            attribute_distributions=distributions,
            top_k_instability=1.0,
        )

    scores = _normalized_scores(competitive)
    probabilities = _softmax(scores, policy.score_temperature)
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    effective_mass = math.exp(entropy) / len(probabilities)
    concentration = max(probabilities)
    uncertainty, coverage, distributions = _attribute_statistics(competitive)
    metadata_sufficiency = sum(coverage.values()) / len(coverage)
    boundary_index = min(policy.top_k_boundary - 1, len(scores) - 1)
    next_index = min(boundary_index + 1, len(scores) - 1)
    boundary_margin = 1.0 if boundary_index == next_index else _margin(
        scores, boundary_index, next_index
    )
    lead_margin = 1.0 if len(scores) == 1 else _margin(scores, 0, 1)
    # Small lead/boundary margins make Top-10 membership/order fragile.
    top_k_instability = 1.0 - (0.35 * lead_margin + 0.65 * boundary_margin)

    return CandidatePoolDiagnostics(
        candidate_count=len(competitive),
        effective_candidate_mass=min(1.0, max(0.0, effective_mass)),
        score_concentration=min(1.0, max(0.0, concentration)),
        lead_margin=lead_margin,
        top_k_boundary_margin=boundary_margin,
        route_disagreement=_route_disagreement(competitive, policy.route_top_k),
        constraint_coverage=_constraint_coverage(state, competitive),
        metadata_sufficiency=metadata_sufficiency,
        attribute_uncertainty=uncertainty,
        attribute_coverage=coverage,
        attribute_distributions=distributions,
        top_k_instability=min(1.0, max(0.0, top_k_instability)),
    )
