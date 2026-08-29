"""Conservative constraint evaluation for validated retrieval candidates.

The module deliberately uses structural access instead of importing Person 1's
state or Person 2's retrieval implementation.  Objects only need to expose the
fields frozen in contract version 0.1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def enum_value(value: object) -> str:
    """Return a serialized enum/literal value without depending on enum types."""

    raw = getattr(value, "value", value)
    return str(raw).lower()


def clamp01(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, numeric))


def active_constraints(state: object) -> tuple[object, ...]:
    constraints = getattr(state, "active_constraints", ()) or ()
    return tuple(
        constraint
        for constraint in constraints
        if enum_value(getattr(constraint, "status", "active")) == "active"
    )


def evidence(candidate: object) -> object:
    return getattr(candidate, "product_evidence", None)


def constraint_outcomes(candidate: object) -> Mapping[object, object]:
    return getattr(evidence(candidate), "constraint_outcomes", {}) or {}


def attribute_values(candidate: object) -> Mapping[object, tuple[object, ...]]:
    return getattr(evidence(candidate), "attribute_values", {}) or {}


def evidence_reliability(candidate: object) -> Mapping[object, object]:
    return getattr(evidence(candidate), "evidence_reliability", {}) or {}


def mapping_lookup(mapping: Mapping[object, Any], attribute: object, default: Any) -> Any:
    """Read a mapping keyed by either Literal strings or Enum members."""

    if attribute in mapping:
        return mapping[attribute]
    target = enum_value(attribute)
    for key, value in mapping.items():
        if enum_value(key) == target:
            return value
    return default


@dataclass(frozen=True)
class ConstraintPolicyConfig:
    hard_contradiction_reliability: float = 0.80
    hard_constraint_confidence: float = 0.80
    match_weight: float = 1.0
    soft_contradiction_weight: float = 0.75
    unknown_penalty: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 <= self.hard_contradiction_reliability <= 1.0:
            raise ValueError("hard_contradiction_reliability must be in [0, 1]")
        if not 0.0 <= self.hard_constraint_confidence <= 1.0:
            raise ValueError("hard_constraint_confidence must be in [0, 1]")
        if min(self.match_weight, self.soft_contradiction_weight, self.unknown_penalty) < 0:
            raise ValueError("constraint weights must be non-negative")


@dataclass(frozen=True)
class ConstraintAssessment:
    eligible: bool
    match_score: float
    soft_contradiction_score: float
    unknown_count: int
    hard_contradictions: tuple[str, ...]
    matched_attributes: tuple[str, ...]


def assess_candidate(
    state: object,
    candidate: object,
    config: ConstraintPolicyConfig | None = None,
) -> ConstraintAssessment:
    """Assess evidence without treating missing metadata as contradiction."""

    policy = config or ConstraintPolicyConfig()
    outcomes = constraint_outcomes(candidate)
    reliability = evidence_reliability(candidate)
    hard_contradictions: list[str] = []
    matched: list[str] = []
    match_score = 0.0
    soft_contradiction_score = 0.0
    unknown_count = 0

    for constraint in active_constraints(state):
        attribute = getattr(constraint, "attribute", "")
        attribute_name = enum_value(attribute)
        outcome = enum_value(mapping_lookup(outcomes, attribute, "unknown"))
        confidence = clamp01(getattr(constraint, "confidence", 0.0))
        strength = enum_value(getattr(constraint, "strength", "soft"))
        evidence_strength = clamp01(mapping_lookup(reliability, attribute, 0.0))

        if outcome == "match":
            matched.append(attribute_name)
            match_score += policy.match_weight * confidence * max(0.25, evidence_strength)
        elif outcome == "contradiction":
            confirmed_hard = (
                strength == "hard"
                and confidence >= policy.hard_constraint_confidence
                and evidence_strength >= policy.hard_contradiction_reliability
            )
            if confirmed_hard:
                hard_contradictions.append(attribute_name)
            else:
                soft_contradiction_score += (
                    policy.soft_contradiction_weight
                    * confidence
                    * max(0.25, evidence_strength)
                )
        else:
            unknown_count += 1

    return ConstraintAssessment(
        eligible=not hard_contradictions,
        match_score=match_score,
        soft_contradiction_score=soft_contradiction_score,
        unknown_count=unknown_count,
        hard_contradictions=tuple(sorted(set(hard_contradictions))),
        matched_attributes=tuple(sorted(set(matched))),
    )


def known_values(candidate: object, attribute: object) -> tuple[object, ...]:
    values = mapping_lookup(attribute_values(candidate), attribute, ())
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def normalized_value(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return repr(value).casefold()


def has_value(candidate: object, attribute: object, value: object) -> bool:
    target = normalized_value(value)
    return any(normalized_value(item) == target for item in known_values(candidate, attribute))


def unique_candidates(candidates: Iterable[object]) -> tuple[object, ...]:
    seen: set[str] = set()
    result: list[object] = []
    for candidate in candidates:
        parent_asin = str(getattr(candidate, "parent_asin", "")).strip()
        if not parent_asin or parent_asin in seen:
            continue
        seen.add(parent_asin)
        result.append(candidate)
    return tuple(result)
