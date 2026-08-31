"""Reproducible constraint-aware ranking and network-free fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
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
    route_rank_weight: float = 0.0
    route_rank_k: float = 60.0
    evidence_phrase_weight: float = 0.0
    evidence_phrase_min_confidence: float = 0.65
    evidence_specificity_weight: float = 0.0
    popularity_weight: float = 0.0
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
            self.route_rank_weight,
            self.evidence_phrase_weight,
            self.evidence_specificity_weight,
            self.popularity_weight,
            self.constraint_match_weight,
            self.profile_weight,
            self.soft_contradiction_penalty,
            self.unknown_penalty,
            self.shown_penalty,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("ranking weights must be non-negative")
        if not math.isfinite(self.route_rank_k) or self.route_rank_k < 0:
            raise ValueError("route_rank_k must be finite and non-negative")
        if not 0.0 <= self.evidence_phrase_min_confidence <= 1.0:
            raise ValueError("evidence_phrase_min_confidence must be in [0, 1]")
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


def _route_rank_score(candidate: object, k: float) -> float:
    """Return route-only reciprocal-rank evidence without structural boosts."""

    ranks = tuple(
        rank
        for field in ("sparse_rank", "dense_rank")
        if math.isfinite(rank := _rank_or_infinity(getattr(candidate, field, None)))
    )
    return sum(1.0 / (k + rank) for rank in ranks)


_EVIDENCE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EVIDENCE_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "i",
        "in", "is", "it", "my", "of", "on", "or", "that", "the", "this",
        "to", "what", "with", "you", "your",
    }
)
_NON_PRODUCT_PHRASES = (
    "options are not quite right",
    "ask me about one specific attribute",
    "still exploring",
    "don't have an additional preference",
    "do not have an additional preference",
)


def _evidence_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in _EVIDENCE_TOKEN_RE.findall(str(value).casefold())
        if len(token) > 1 and token not in _EVIDENCE_STOP_WORDS
    )


def _constraint_phrases(state: object, minimum_confidence: float) -> tuple[str, ...]:
    phrases: list[str] = []
    for constraint in getattr(state, "active_constraints", ()) or ():
        if str(getattr(constraint, "status", "active")).casefold() != "active":
            continue
        if _finite(getattr(constraint, "confidence", 0.0)) < minimum_confidence:
            continue
        polarity = getattr(getattr(constraint, "polarity", "include"), "value", None)
        if str(polarity or getattr(constraint, "polarity", "include")).casefold() != "include":
            continue
        raw = getattr(constraint, "normalized_value", getattr(constraint, "value", ""))
        phrase = " ".join(str(raw).casefold().split())
        if not phrase or any(marker in phrase for marker in _NON_PRODUCT_PHRASES):
            continue
        if _evidence_tokens(phrase):
            phrases.append(phrase)
    return tuple(dict.fromkeys(phrases))


def _candidate_evidence_text(candidate: object) -> str:
    evidence = getattr(candidate, "product_evidence", None)
    parts = list(getattr(evidence, "supporting_snippets", ()) or ())
    for values in (getattr(evidence, "attribute_values", {}) or {}).values():
        if isinstance(values, (str, bytes)):
            parts.append(str(values))
        else:
            parts.extend(str(value) for value in (values or ()))
    return " ".join(" ".join(str(part).casefold().split()) for part in parts)


def evidence_phrase_scores(
    state: object,
    candidates: Sequence[object],
    minimum_confidence: float = 0.65,
) -> tuple[float, ...]:
    """Score rare token, ordered phrase, and exact phrase evidence in-pool."""

    phrases = _constraint_phrases(state, minimum_confidence)
    if not candidates or not phrases:
        return tuple(0.0 for _ in candidates)
    query_tokens = tuple(
        dict.fromkeys(token for phrase in phrases for token in _evidence_tokens(phrase))
    )
    texts = tuple(_candidate_evidence_text(candidate) for candidate in candidates)
    token_sets = tuple(set(_evidence_tokens(text)) for text in texts)
    frequency = Counter(token for values in token_sets for token in values)
    idf = {
        token: math.log((len(candidates) + 1) / (frequency[token] + 1)) + 1.0
        for token in query_tokens
    }
    token_total = sum(idf.values()) or 1.0
    phrase_token_total = sum(len(_evidence_tokens(phrase)) for phrase in phrases) or 1
    scores: list[float] = []
    for text, present in zip(texts, token_sets):
        token_coverage = sum(
            idf[token] for token in query_tokens if token in present
        ) / token_total
        matched_bigrams = 0
        total_bigrams = 0
        exact_tokens = 0
        for phrase in phrases:
            phrase_tokens = _evidence_tokens(phrase)
            bigrams = tuple(zip(phrase_tokens, phrase_tokens[1:]))
            total_bigrams += len(bigrams)
            matched_bigrams += sum(
                f"{left} {right}" in text for left, right in bigrams
            )
            if len(phrase_tokens) >= 2 and phrase in text:
                exact_tokens += len(phrase_tokens)
        bigram_coverage = matched_bigrams / max(1, total_bigrams)
        exact_coverage = exact_tokens / phrase_token_total
        scores.append(
            clamp01(
                0.55 * token_coverage
                + 0.25 * bigram_coverage
                + 0.20 * exact_coverage
            )
        )
    return tuple(scores)


def evidence_specificity_scores(candidates: Sequence[object]) -> tuple[float, ...]:
    """Prefer bounded, information-rich evidence when relevance otherwise ties."""

    raw: list[float] = []
    for candidate in candidates:
        evidence = getattr(candidate, "product_evidence", None)
        text = _candidate_evidence_text(candidate)
        unique_tokens = len(set(_evidence_tokens(text)))
        matched_fields = len(set(getattr(evidence, "matched_fields", ()) or ()))
        known_attributes = sum(
            bool(values)
            for values in (getattr(evidence, "attribute_values", {}) or {}).values()
        )
        raw.append(
            math.log1p(min(unique_tokens, 500))
            + 0.15 * min(matched_fields, 6)
            + 0.08 * min(known_attributes, 10)
        )
    return normalized_score_magnitudes(tuple(raw))


def _route_detail_number(candidate: object, field: str) -> float:
    evidence = getattr(candidate, "product_evidence", None)
    details = getattr(evidence, "route_details", {}) or {}
    return max(0.0, _finite(details.get(field, 0.0)))


def popularity_scores(candidates: Sequence[object]) -> tuple[float, ...]:
    """Return a bounded purchase-likelihood prior from public catalog counts."""

    return normalized_score_magnitudes(
        tuple(
            math.log1p(_route_detail_number(candidate, "rating_number"))
            for candidate in candidates
        )
    )


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
        route_rank = normalized_score_magnitudes(
            tuple(_route_rank_score(item, self.config.route_rank_k) for item in unique)
        )
        phrase_evidence = (
            evidence_phrase_scores(
                state,
                unique,
                self.config.evidence_phrase_min_confidence,
            )
            if self.config.evidence_phrase_weight > 0.0
            else tuple(0.0 for _ in unique)
        )
        evidence_specificity = (
            evidence_specificity_scores(unique)
            if self.config.evidence_specificity_weight > 0.0
            else tuple(0.0 for _ in unique)
        )
        popularity = (
            popularity_scores(unique)
            if self.config.popularity_weight > 0.0
            else tuple(0.0 for _ in unique)
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
                + self.config.route_rank_weight * route_rank[index]
                + self.config.evidence_phrase_weight * phrase_evidence[index]
                + self.config.evidence_specificity_weight
                * evidence_specificity[index]
                + self.config.popularity_weight * popularity[index]
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
