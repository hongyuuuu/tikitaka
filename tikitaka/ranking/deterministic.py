"""Reproducible constraint-aware ranking and network-free fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tikitaka.contracts import Usage

from .constraints import (
    ConstraintAssessment,
    ConstraintPolicyConfig,
    assess_candidate,
    clamp01,
    unique_candidates,
)


# Kept as a public compatibility name for the pre-contract implementation.
UsageRecord = Usage


@dataclass(frozen=True)
class DeterministicRankerConfig:
    fused_weight: float = 0.52
    structural_weight: float = 0.16
    route_agreement_weight: float = 0.08
    constraint_match_weight: float = 0.20
    profile_weight: float = 0.0
    soft_contradiction_penalty: float = 0.22
    unknown_penalty: float = 0.015
    shown_penalty: float = 0.40
    exclude_confirmed_hard_contradictions: bool = True
    exclude_shown_when_unseen_available: bool = True
    hard_contradiction_reliability: float = 0.80
    hard_constraint_confidence: float = 0.80

    def __post_init__(self) -> None:
        weights = (
            self.fused_weight,
            self.structural_weight,
            self.route_agreement_weight,
            self.constraint_match_weight,
            self.profile_weight,
            self.soft_contradiction_penalty,
            self.unknown_penalty,
            self.shown_penalty,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("ranking weights must be non-negative")
        if not 0.0 <= self.hard_contradiction_reliability <= 1.0:
            raise ValueError("hard_contradiction_reliability must be in [0, 1]")
        if not 0.0 <= self.hard_constraint_confidence <= 1.0:
            raise ValueError("hard_constraint_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: object
    score: float
    assessment: ConstraintAssessment
    shown_in_current_intent: bool

    @property
    def parent_asin(self) -> str:
        return str(getattr(self.candidate, "parent_asin"))


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalized_score_magnitudes(values: Sequence[float]) -> tuple[float, ...]:
    """Bound score magnitudes without turning tiny pool-local gaps into 0/1 gaps.

    Retrieval scores are already comparable within a route.  Dividing
    non-negative scores by the largest magnitude preserves that evidence,
    whereas pool-local min/max scaling can make 0.501 look categorically better
    than 0.500.  Signed scores are mapped around a neutral 0.5 instead.
    """

    if not values:
        return ()
    finite = tuple(_finite(value) for value in values)
    low = min(finite)
    high = max(finite)
    if low >= 0.0:
        if math.isclose(high, 0.0):
            return tuple(0.5 for _ in finite)
        return tuple(clamp01(value / high) for value in finite)
    scale = max(abs(low), abs(high))
    if math.isclose(scale, 0.0):
        return tuple(0.5 for _ in finite)
    return tuple(clamp01(0.5 + 0.5 * value / scale) for value in finite)


def _route_agreement(candidate: object) -> float:
    present = sum(
        getattr(candidate, field, None) is not None
        for field in ("sparse_rank", "dense_rank")
    )
    structural = abs(_finite(getattr(candidate, "structural_score", 0.0))) > 0.0
    return min(1.0, (present + int(structural)) / 3.0)


def _profile_contribution(candidate: object) -> float:
    product_evidence = getattr(candidate, "product_evidence", None)
    return clamp01(getattr(product_evidence, "profile_contribution", 0.0))


class DeterministicRanker:
    """Score and order only the validated candidate objects supplied to it."""

    def __init__(
        self,
        config: DeterministicRankerConfig | None = None,
        usage_type: type = Usage,
    ) -> None:
        self.config = config or DeterministicRankerConfig()
        self.usage_type = usage_type
        self.constraint_config = ConstraintPolicyConfig(
            hard_contradiction_reliability=self.config.hard_contradiction_reliability,
            hard_constraint_confidence=self.config.hard_constraint_confidence,
        )

    def rank_candidates(
        self,
        state: object,
        candidates: Sequence[object],
    ) -> tuple[ScoredCandidate, ...]:
        unique = unique_candidates(candidates)
        if not unique:
            return ()

        fused = normalized_score_magnitudes(
            tuple(_finite(getattr(item, "fused_score", 0.0)) for item in unique)
        )
        structural = normalized_score_magnitudes(
            tuple(_finite(getattr(item, "structural_score", 0.0)) for item in unique)
        )
        shown_ids = {
            str(item) for item in (getattr(state, "shown_product_ids", ()) or ())
        }
        scored: list[ScoredCandidate] = []

        for index, candidate in enumerate(unique):
            assessment = assess_candidate(state, candidate, self.constraint_config)
            if self.config.exclude_confirmed_hard_contradictions and not assessment.eligible:
                continue
            is_shown = str(getattr(candidate, "parent_asin")) in shown_ids
            score = (
                self.config.fused_weight * fused[index]
                + self.config.structural_weight * structural[index]
                + self.config.route_agreement_weight * _route_agreement(candidate)
                + self.config.constraint_match_weight * assessment.match_score
                + self.config.profile_weight * _profile_contribution(candidate)
                - self.config.soft_contradiction_penalty
                * assessment.soft_contradiction_score
                - self.config.unknown_penalty * assessment.unknown_count
                - self.config.shown_penalty * int(is_shown)
            )
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    score=score,
                    assessment=assessment,
                    shown_in_current_intent=is_shown,
                )
            )

        scored.sort(
            key=lambda item: (
                -item.score,
                -_finite(getattr(item.candidate, "fused_score", 0.0)),
                -_finite(getattr(item.candidate, "structural_score", 0.0)),
                _best_route_rank(item.candidate),
                _rank_or_infinity(getattr(item.candidate, "sparse_rank", None)),
                _rank_or_infinity(getattr(item.candidate, "dense_rank", None)),
                item.parent_asin,
            )
        )
        return tuple(scored)

    def select_candidates(
        self,
        state: object,
        candidates: Sequence[object],
        limit: int,
    ) -> tuple[ScoredCandidate, ...]:
        """Prefer unseen products and use shown products only as backfill."""

        if limit < 0:
            raise ValueError("limit must be non-negative")
        ranked = self.rank_candidates(state, candidates)
        if not self.config.exclude_shown_when_unseen_available:
            return ranked[:limit]
        unseen = [item for item in ranked if not item.shown_in_current_intent]
        shown = [item for item in ranked if item.shown_in_current_intent]
        return tuple((unseen + shown)[:limit])

    def rank(
        self,
        state: object,
        candidates: Sequence[object],
        top_k: int,
    ) -> tuple[list[str], UsageRecord]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        ranked = self.select_candidates(state, candidates, top_k)
        return [item.parent_asin for item in ranked[:top_k]], self.usage_type(
            route="deterministic"
        )


def _rank_or_infinity(rank: object) -> float:
    value = _finite(rank, math.inf)
    return value if value > 0 else math.inf


def _best_route_rank(candidate: object) -> float:
    return min(
        _rank_or_infinity(getattr(candidate, "sparse_rank", None)),
        _rank_or_infinity(getattr(candidate, "dense_rank", None)),
    )
