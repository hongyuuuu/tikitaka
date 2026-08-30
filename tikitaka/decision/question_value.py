"""Expected rank-weighted Top-10 change for clarification attributes."""

from __future__ import annotations

import math
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


HIGHEST_VALUE_SELECTION = "highest_value"
CONTRACT_ORDER_SELECTION = "contract_order"
QUESTION_SELECTION_STRATEGIES = frozenset(
    {HIGHEST_VALUE_SELECTION, CONTRACT_ORDER_SELECTION}
)


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
    candidate_probability_temperature: float = 0.18

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
        if self.candidate_probability_temperature <= 0.0:
            raise ValueError("candidate_probability_temperature must be positive")


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


def _ordered_question_values(
    values: Sequence[AttributeQuestionValue],
    selection_strategy: str,
) -> tuple[AttributeQuestionValue, ...]:
    if selection_strategy not in QUESTION_SELECTION_STRATEGIES:
        raise ValueError(
            "selection_strategy must be one of "
            f"{sorted(QUESTION_SELECTION_STRATEGIES)}"
        )
    if selection_strategy == CONTRACT_ORDER_SELECTION:
        return tuple(
            sorted(
                values,
                key=lambda item: ALLOWED_ATTRIBUTES.index(item.attribute),
            )
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                -item.expected_information_gain,
                -item.coverage,
                ALLOWED_ATTRIBUTES.index(item.attribute),
            ),
        )
    )


def _active_answered_attributes(state: object, confidence: float) -> set[str]:
    return {
        enum_value(getattr(constraint, "attribute", ""))
        for constraint in active_constraints(state)
        if float(getattr(constraint, "confidence", 0.0)) >= confidence
    }


def _string_set(values: object) -> set[str]:
    return {enum_value(value) for value in (values or ())}


def _candidate_probabilities(
    scored: Sequence[ScoredCandidate], temperature: float
) -> Mapping[str, float]:
    if not scored:
        return {}
    high = max(item.score for item in scored)
    weights = {
        item.parent_asin: math.exp((item.score - high) / temperature)
        for item in scored
    }
    total = sum(weights.values()) or 1.0
    return {parent_asin: weight / total for parent_asin, weight in weights.items()}


def _reciprocal_rank(position: int | None, top_k: int) -> float:
    if position is None or position >= top_k:
        return 0.0
    return 1.0 / (position + 1.0)


def _ranking_change(
    base: Sequence[str],
    branch: Sequence[str],
    top_k: int,
    membership_weight: float,
    order_weight: float,
    candidate_probabilities: Mapping[str, float],
) -> float:
    base_set = set(base[:top_k])
    branch_set = set(branch[:top_k])
    membership_change = sum(
        candidate_probabilities.get(parent_asin, 0.0)
        for parent_asin in base_set ^ branch_set
    )
    base_positions = {parent_asin: rank for rank, parent_asin in enumerate(base)}
    branch_positions = {parent_asin: rank for rank, parent_asin in enumerate(branch)}
    order_change = sum(
        candidate_probabilities.get(parent_asin, 0.0)
        * abs(
            _reciprocal_rank(base_positions.get(parent_asin), top_k)
            - _reciprocal_rank(branch_positions.get(parent_asin), top_k)
        )
        for parent_asin in set(base) | set(branch)
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
        selection_strategy: str = HIGHEST_VALUE_SELECTION,
    ) -> None:
        self.ranker = ranker or DeterministicRanker()
        self.config = config or QuestionValueConfig()
        if selection_strategy not in QUESTION_SELECTION_STRATEGIES:
            raise ValueError(
                "selection_strategy must be one of "
                f"{sorted(QUESTION_SELECTION_STRATEGIES)}"
            )
        self.selection_strategy = selection_strategy

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
        candidate_probabilities = _candidate_probabilities(
            scored, self.config.candidate_probability_temperature
        )
        answered = _active_answered_attributes(
            state, self.config.confident_answer_threshold
        )
        no_preference = _string_set(getattr(state, "no_preference", ()))
        asked = _string_set(getattr(state, "asked_attributes", ()))
        exhausted = _string_set(getattr(state, "exhausted_attributes", ()))
        revalidation = {
            enum_value(getattr(item, "attribute", ""))
            for item in (getattr(state, "revalidation_constraints", ()) or ())
        }
        values: list[AttributeQuestionValue] = []
        turn_discount = 0.5 + 0.5 * ((10 - turn) / 9.0)

        for attribute in ALLOWED_ATTRIBUTES:
            if attribute in no_preference or attribute in exhausted:
                continue
            if attribute in asked and attribute not in revalidation:
                continue
            if attribute in answered and attribute not in revalidation:
                continue
            value_mass: dict[str, float] = {}
            known_mass = 0.0
            for item in scored:
                normalized = {
                    normalized_value(value)
                    for value in known_values(item.candidate, attribute)
                }
                if not normalized:
                    continue
                candidate_mass = candidate_probabilities[item.parent_asin]
                known_mass += candidate_mass
                share = 1.0 / len(normalized)
                for value in normalized:
                    value_mass[value] = value_mass.get(value, 0.0) + candidate_mass * share
            coverage = known_mass
            if coverage < self.config.minimum_attribute_coverage:
                continue
            branches = sorted(
                value_mass.items(), key=lambda item: (-item[1], item[0])
            )[: self.config.maximum_branches]
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
                    candidate_probabilities,
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
                    distinct_values=len(value_mass),
                    branch_probabilities=probabilities,
                )
            )

        ordered = _ordered_question_values(values, self.selection_strategy)
        if not ordered:
            return QuestionValueResult(None, 0.0, ())
        return QuestionValueResult(
            best_attribute=ordered[0].attribute,
            expected_information_gain=ordered[0].expected_information_gain,
            values=ordered,
        )
