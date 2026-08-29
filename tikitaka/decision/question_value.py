"""Expected rank-weighted Top-10 change for clarification attributes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from tikitaka.ranking.constraints import (
    active_constraints,
    enum_value,
    has_value,
    known_values,
    normalized_value,
)
from tikitaka.ranking.deterministic import DeterministicRanker, ScoredCandidate

from .diagnostics import ALLOWED_ATTRIBUTES


@dataclass(frozen=True)
class QuestionValueConfig:
    competitive_limit: int = 50
    ranking_top_k: int = 10
    minimum_attribute_coverage: float = 0.35
    minimum_distinct_values: int = 2
    maximum_branches: int = 8
    confident_answer_threshold: float = 0.70
    temporary_match_boost: float = 0.35
    temporary_mismatch_penalty: float = 0.25
    membership_weight: float = 0.75
    order_weight: float = 0.25

    def __post_init__(self) -> None:
        if min(self.competitive_limit, self.ranking_top_k, self.maximum_branches) <= 0:
            raise ValueError("question-value limits must be positive")
        bounded = (
            self.minimum_attribute_coverage,
            self.confident_answer_threshold,
            self.membership_weight,
            self.order_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("question-value proportions must be in [0, 1]")
        if not math.isclose(self.membership_weight + self.order_weight, 1.0):
            raise ValueError("membership and order weights must sum to 1")


@dataclass(frozen=True)
class AttributeQuestionValue:
    attribute: str
    expected_information_gain: float
    coverage: float
    distinct_values: int
    branch_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class QuestionValueResult:
    best_attribute: str | None
    expected_information_gain: float
    values: tuple[AttributeQuestionValue, ...]


def _active_answered_attributes(state: object, confidence: float) -> set[str]:
    return {
        enum_value(getattr(constraint, "attribute", ""))
        for constraint in active_constraints(state)
        if float(getattr(constraint, "confidence", 0.0)) >= confidence
    }


def _string_set(values: object) -> set[str]:
    return {enum_value(value) for value in (values or ())}


def _rank_weights(ids: Sequence[str], top_k: int) -> Mapping[str, float]:
    limited = ids[:top_k]
    raw = {parent_asin: 1.0 / math.log2(rank + 2.0) for rank, parent_asin in enumerate(limited)}
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}


def _ranking_change(
    base: Sequence[str],
    branch: Sequence[str],
    top_k: int,
    membership_weight: float,
    order_weight: float,
) -> float:
    base_set = set(base[:top_k])
    branch_set = set(branch[:top_k])
    union = base_set | branch_set
    membership_change = 0.0 if not union else 1.0 - len(base_set & branch_set) / len(union)
    base_weights = _rank_weights(base, top_k)
    branch_weights = _rank_weights(branch, top_k)
    order_change = 0.5 * sum(
        abs(base_weights.get(parent_asin, 0.0) - branch_weights.get(parent_asin, 0.0))
        for parent_asin in union
    )
    return min(
        1.0,
        max(0.0, membership_weight * membership_change + order_weight * order_change),
    )


def _branch_order(
    scored: Sequence[ScoredCandidate],
    attribute: str,
    value: str,
    config: QuestionValueConfig,
) -> list[str]:
    adjusted: list[tuple[float, float, str]] = []
    for item in scored:
        values = known_values(item.candidate, attribute)
        if has_value(item.candidate, attribute, value):
            adjustment = config.temporary_match_boost
        elif values:
            adjustment = -config.temporary_mismatch_penalty
        else:
            adjustment = 0.0
        adjusted.append((item.score + adjustment, item.score, item.parent_asin))
    adjusted.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))
    return [parent_asin for _, _, parent_asin in adjusted]


class QuestionValueEstimator:
    def __init__(
        self,
        ranker: DeterministicRanker | None = None,
        config: QuestionValueConfig | None = None,
    ) -> None:
        self.ranker = ranker or DeterministicRanker()
        self.config = config or QuestionValueConfig()

    def estimate(
        self,
        state: object,
        candidates: Sequence[object],
        turn: int,
    ) -> QuestionValueResult:
        if not 1 <= turn <= 10:
            raise ValueError("turn must be in the official 1-to-10 range")
        scored = self.ranker.rank_candidates(state, candidates)[: self.config.competitive_limit]
        if not scored or turn == 10:
            return QuestionValueResult(None, 0.0, ())

        base_ids = [item.parent_asin for item in scored]
        answered = _active_answered_attributes(
            state, self.config.confident_answer_threshold
        )
        no_preference = _string_set(getattr(state, "no_preference", ()))
        asked = _string_set(getattr(state, "asked_attributes", ()))
        revalidation = {
            enum_value(getattr(item, "attribute", ""))
            for item in (getattr(state, "revalidation_constraints", ()) or ())
        }
        values: list[AttributeQuestionValue] = []
        turn_discount = 0.5 + 0.5 * ((10 - turn) / 9.0)

        for attribute in ALLOWED_ATTRIBUTES:
            if attribute in no_preference or attribute in asked:
                continue
            if attribute in answered and attribute not in revalidation:
                continue
            counter: Counter[str] = Counter()
            known_count = 0
            for item in scored:
                normalized = {
                    normalized_value(value)
                    for value in known_values(item.candidate, attribute)
                }
                if not normalized:
                    continue
                known_count += 1
                share = 1.0 / len(normalized)
                for value in normalized:
                    counter[value] += share
            coverage = known_count / len(scored)
            if coverage < self.config.minimum_attribute_coverage:
                continue
            branches = counter.most_common(self.config.maximum_branches)
            if len(branches) < self.config.minimum_distinct_values:
                continue
            branch_total = sum(amount for _, amount in branches)
            probabilities = {
                value: amount / branch_total for value, amount in branches
            }
            expected_change = 0.0
            for value, probability in probabilities.items():
                branch_ids = _branch_order(scored, attribute, value, self.config)
                expected_change += probability * _ranking_change(
                    base_ids,
                    branch_ids,
                    self.config.ranking_top_k,
                    self.config.membership_weight,
                    self.config.order_weight,
                )
            information_gain = min(
                1.0,
                max(0.0, expected_change * coverage * turn_discount),
            )
            values.append(
                AttributeQuestionValue(
                    attribute=attribute,
                    expected_information_gain=information_gain,
                    coverage=coverage,
                    distinct_values=len(counter),
                    branch_probabilities=probabilities,
                )
            )

        values.sort(
            key=lambda item: (
                -item.expected_information_gain,
                -item.coverage,
                ALLOWED_ATTRIBUTES.index(item.attribute),
            )
        )
        if not values:
            return QuestionValueResult(None, 0.0, ())
        return QuestionValueResult(
            best_attribute=values[0].attribute,
            expected_information_gain=values[0].expected_information_gain,
            values=tuple(values),
        )
