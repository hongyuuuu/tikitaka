"""Strict, immutable shared domain records.

Trusted internal constructors reject invalid normalized values. Parsers at an
untrusted boundary may use :func:`clamp_unit_interval` before construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, TypeVar


class _ClosedValue(str, Enum):
    def __str__(self) -> str:
        return self.value


class Attribute(_ClosedValue):
    CATEGORY = "category"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    STYLE = "style"
    BRAND = "brand"
    BUDGET = "budget"
    FEATURE = "feature"
    USE_CASE = "use_case"
    OTHER = "other"


class ConstraintPolarity(_ClosedValue):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class ConstraintStrength(_ClosedValue):
    HARD = "hard"
    SOFT = "soft"


class ConstraintStatus(_ClosedValue):
    ACTIVE = "active"
    REPLACED = "replaced"
    RETRACTED = "retracted"
    NEEDS_REVALIDATION = "needs_revalidation"


class StateOperationKind(_ClosedValue):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    EXCLUDE = "exclude"
    NO_PREFERENCE = "no_preference"
    RESET = "reset"


class OperationScope(_ClosedValue):
    ATTRIBUTE = "attribute"
    CONVERSATION = "conversation"
    INTENT = "intent"


class InferredMode(_ClosedValue):
    BUYING = "buying"
    BROWSING = "browsing"
    UNKNOWN = "unknown"


class TurnAction(_ClosedValue):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"


class EvidenceOutcome(_ClosedValue):
    MATCH = "match"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"


class RoutePolicy(_ClosedValue):
    AUTO = "auto"
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"


class DecisionReasonCode(_ClosedValue):
    FINAL_TURN = "final_turn"
    VALUABLE_CLARIFICATION = "valuable_clarification"
    LOW_QUESTION_VALUE = "low_question_value"
    NO_ELIGIBLE_ATTRIBUTE = "no_eligible_attribute"
    RANKING_STABLE = "ranking_stable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    COMPONENT_FALLBACK = "component_fallback"


EnumT = TypeVar("EnumT", bound=Enum)


def _enum(value: object, enum_type: type[EnumT], name: str) -> EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown {name}: {value!r}") from error


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _unit_interval(value: object, name: str) -> float:
    converted = _finite_number(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return converted


def clamp_unit_interval(value: object) -> float:
    """Normalize an untrusted finite numeric value into ``[0.0, 1.0]``."""

    return min(1.0, max(0.0, _finite_number(value, "value")))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: Mapping[object, object], name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _attribute_mapping(
    value: Mapping[object, object],
    name: str,
) -> Mapping[Attribute, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType({_enum(key, Attribute, "attribute"): item for key, item in value.items()})


@dataclass(frozen=True)
class Constraint:
    attribute: Attribute
    value: object
    normalized_value: object
    polarity: ConstraintPolarity
    strength: ConstraintStrength
    source_turn: int
    confidence: float
    intent_version: int
    status: ConstraintStatus = ConstraintStatus.ACTIVE
    category_dependent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "attribute", _enum(self.attribute, Attribute, "attribute"))
        object.__setattr__(self, "polarity", _enum(self.polarity, ConstraintPolarity, "polarity"))
        object.__setattr__(self, "strength", _enum(self.strength, ConstraintStrength, "strength"))
        object.__setattr__(self, "status", _enum(self.status, ConstraintStatus, "status"))
        if isinstance(self.source_turn, bool) or not isinstance(self.source_turn, int) or not 1 <= self.source_turn <= 10:
            raise ValueError("source_turn must be between 1 and 10")
        _positive_integer(self.intent_version, "intent_version")
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        if not isinstance(self.category_dependent, bool):
            raise TypeError("category_dependent must be a bool")


@dataclass(frozen=True)
class StateOperation:
    operation: StateOperationKind
    attribute: Attribute | None
    old_value: object | None
    new_value: object | None
    scope: OperationScope
    polarity: ConstraintPolarity | None
    strength: ConstraintStrength | None
    confidence: float | None

    def __post_init__(self) -> None:
        operation = _enum(self.operation, StateOperationKind, "operation")
        scope = _enum(self.scope, OperationScope, "scope")
        attribute = None if self.attribute is None else _enum(self.attribute, Attribute, "attribute")
        polarity = None if self.polarity is None else _enum(self.polarity, ConstraintPolarity, "polarity")
        strength = None if self.strength is None else _enum(self.strength, ConstraintStrength, "strength")
        confidence = None if self.confidence is None else _unit_interval(self.confidence, "confidence")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "confidence", confidence)

        if operation is StateOperationKind.RESET:
            if scope not in (OperationScope.CONVERSATION, OperationScope.INTENT):
                raise ValueError("reset requires conversation or intent scope")
            if any(value is not None for value in (attribute, self.old_value, self.new_value, polarity, strength, confidence)):
                raise ValueError("reset contains fields that must be None")
            return
        if scope is not OperationScope.ATTRIBUTE:
            raise ValueError(f"{operation.value} requires attribute scope")
        if operation in (StateOperationKind.ADD, StateOperationKind.EXCLUDE):
            if attribute is None or self.new_value is None or strength is None or confidence is None or polarity is None:
                raise ValueError(f"{operation.value} is missing a required field")
            if self.old_value is not None:
                raise ValueError(f"{operation.value} requires old_value to be None")
            if operation is StateOperationKind.EXCLUDE:
                object.__setattr__(self, "polarity", ConstraintPolarity.EXCLUDE)
        elif operation is StateOperationKind.REPLACE:
            if any(value is None for value in (attribute, self.old_value, self.new_value, polarity, strength, confidence)):
                raise ValueError("replace is missing a required field")
        elif operation is StateOperationKind.REMOVE:
            if attribute is None and self.old_value is None:
                raise ValueError("remove requires attribute or old_value")
            if any(value is not None for value in (self.new_value, polarity, strength, confidence)):
                raise ValueError("remove contains fields that must be None")
        elif operation is StateOperationKind.NO_PREFERENCE:
            if attribute is None:
                raise ValueError("no_preference requires attribute")
            if any(value is not None for value in (self.old_value, self.new_value, polarity, strength, confidence)):
                raise ValueError("no_preference contains fields that must be None")


@dataclass(frozen=True)
class StateDelta:
    inferred_mode: InferredMode
    mode_confidence: float
    operations: tuple[StateOperation, ...]
    generality: float
    rejected_operations: int
    schema_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "inferred_mode", _enum(self.inferred_mode, InferredMode, "mode"))
        object.__setattr__(self, "mode_confidence", _unit_interval(self.mode_confidence, "mode_confidence"))
        object.__setattr__(self, "generality", _unit_interval(self.generality, "generality"))
        object.__setattr__(self, "operations", tuple(self.operations))
        if not all(isinstance(item, StateOperation) for item in self.operations):
            raise TypeError("operations must contain StateOperation values")
        _non_negative_integer(self.rejected_operations, "rejected_operations")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")


@dataclass(frozen=True)
class ProfileBias:
    terms: tuple[str, ...] = ()
    weight: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(str(term) for term in self.terms))
        object.__setattr__(self, "weight", _unit_interval(self.weight, "weight"))


@dataclass(frozen=True)
class SearchPlan:
    text_query: str
    must_terms: tuple[str, ...]
    should_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]
    filters: Mapping[str, object]
    attribute_values: Mapping[Attribute, tuple[object, ...]]
    mode: InferredMode
    intent_version: int
    revalidation_flags: frozenset[Attribute]
    no_preference: frozenset[Attribute]
    profile_bias: ProfileBias
    route_policy: RoutePolicy = RoutePolicy.AUTO
    embedding_route_id: str | None = None
    index_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("must_terms", "should_terms", "exclude_terms"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        object.__setattr__(self, "filters", _mapping(self.filters, "filters"))
        values = _attribute_mapping(self.attribute_values, "attribute_values")
        object.__setattr__(self, "attribute_values", MappingProxyType({key: tuple(item) for key, item in values.items()}))
        object.__setattr__(self, "mode", _enum(self.mode, InferredMode, "mode"))
        object.__setattr__(self, "route_policy", _enum(self.route_policy, RoutePolicy, "route_policy"))
        _positive_integer(self.intent_version, "intent_version")
        object.__setattr__(self, "revalidation_flags", frozenset(_enum(item, Attribute, "attribute") for item in self.revalidation_flags))
        object.__setattr__(self, "no_preference", frozenset(_enum(item, Attribute, "attribute") for item in self.no_preference))
        if not isinstance(self.profile_bias, ProfileBias):
            raise TypeError("profile_bias must be a ProfileBias")
        paired = (self.embedding_route_id is None) == (self.index_id is None)
        if not paired:
            raise ValueError("embedding_route_id and index_id must be set together")
        if self.route_policy in (RoutePolicy.DENSE, RoutePolicy.HYBRID) and self.embedding_route_id is None:
            raise ValueError("dense and hybrid routes require embedding and index identities")


@dataclass(frozen=True)
class ProductEvidence:
    matched_fields: tuple[str, ...]
    supporting_snippets: tuple[str, ...]
    constraint_outcomes: Mapping[Attribute, EvidenceOutcome]
    attribute_values: Mapping[Attribute, tuple[object, ...]]
    evidence_reliability: Mapping[Attribute, float]
    unknown_fields: tuple[str, ...]
    route_details: Mapping[str, object]
    profile_contribution: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_fields", tuple(str(item) for item in self.matched_fields))
        object.__setattr__(self, "supporting_snippets", tuple(str(item) for item in self.supporting_snippets))
        outcomes = _attribute_mapping(self.constraint_outcomes, "constraint_outcomes")
        object.__setattr__(self, "constraint_outcomes", MappingProxyType({key: _enum(item, EvidenceOutcome, "evidence outcome") for key, item in outcomes.items()}))
        values = _attribute_mapping(self.attribute_values, "attribute_values")
        object.__setattr__(self, "attribute_values", MappingProxyType({key: tuple(item) for key, item in values.items()}))
        reliability = _attribute_mapping(self.evidence_reliability, "evidence_reliability")
        object.__setattr__(self, "evidence_reliability", MappingProxyType({key: _unit_interval(item, "evidence reliability") for key, item in reliability.items()}))
        object.__setattr__(self, "unknown_fields", tuple(str(item) for item in self.unknown_fields))
        object.__setattr__(self, "route_details", _mapping(self.route_details, "route_details"))
        object.__setattr__(self, "profile_contribution", _finite_number(self.profile_contribution, "profile_contribution"))


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    product_evidence: ProductEvidence
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    structural_score: float
    fused_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.parent_asin, str) or not self.parent_asin:
            raise ValueError("parent_asin must be a non-empty catalog ID")
        if not isinstance(self.product_evidence, ProductEvidence):
            raise TypeError("product_evidence must be ProductEvidence")
        for name in ("sparse_rank", "dense_rank"):
            value = getattr(self, name)
            if value is not None:
                _positive_integer(value, name)
        for name in ("sparse_score", "dense_score"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(value, name))
        object.__setattr__(self, "structural_score", _finite_number(self.structural_score, "structural_score"))
        object.__setattr__(self, "fused_score", _finite_number(self.fused_score, "fused_score"))

    def retrieval_sort_key(self) -> tuple[float, float, float, float, float, str]:
        infinity = math.inf
        route_ranks = [rank for rank in (self.sparse_rank, self.dense_rank) if rank is not None]
        return (
            -self.fused_score,
            -self.structural_score,
            min(route_ranks, default=infinity),
            self.sparse_rank if self.sparse_rank is not None else infinity,
            self.dense_rank if self.dense_rank is not None else infinity,
            self.parent_asin,
        )


@dataclass(frozen=True)
class TurnDecision:
    action: TurnAction
    ask_attribute: Attribute | None
    reason_code: DecisionReasonCode
    reason: str
    expected_information_gain: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _enum(self.action, TurnAction, "action"))
        object.__setattr__(self, "reason_code", _enum(self.reason_code, DecisionReasonCode, "decision reason code"))
        attribute = None if self.ask_attribute is None else _enum(self.ask_attribute, Attribute, "attribute")
        object.__setattr__(self, "ask_attribute", attribute)
        object.__setattr__(self, "expected_information_gain", _unit_interval(self.expected_information_gain, "expected_information_gain"))
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if self.action is TurnAction.CLARIFY and attribute is None:
            raise ValueError("clarify requires ask_attribute")
        if self.action is TurnAction.RECOMMEND and attribute is not None:
            raise ValueError("recommend requires ask_attribute to be None")


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    repairs: int = 0
    latency_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    reasoning_level: str | None = None
    estimated_cost: float | None = None
    cost_currency: str = "USD"
    route: str | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "calls", "repairs"):
            _non_negative_integer(getattr(self, name), name)
        if self.repairs > self.calls:
            raise ValueError("repairs cannot exceed calls")
        object.__setattr__(self, "latency_ms", _finite_number(self.latency_ms, "latency_ms"))
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.estimated_cost is not None:
            cost = _finite_number(self.estimated_cost, "estimated_cost")
            if cost < 0:
                raise ValueError("estimated_cost cannot be negative")
            object.__setattr__(self, "estimated_cost", cost)
        if not isinstance(self.cost_currency, str) or len(self.cost_currency) != 3 or not self.cost_currency.isalpha():
            raise ValueError("cost_currency must be a three-letter ISO-4217 code")
        object.__setattr__(self, "cost_currency", self.cost_currency.upper())
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a bool")
        if self.cache_hit and any((self.prompt_tokens, self.completion_tokens, self.reasoning_tokens, self.calls, self.repairs, self.latency_ms, self.estimated_cost or 0.0)):
            raise ValueError("a cache hit cannot add calls, usage, latency, or cost")


@dataclass(frozen=True)
class IndexManifest:
    """Immutable identity and provenance for a product embedding index."""

    index_id: str
    catalog_checksum: str
    catalog_row_count: int
    ordered_id_checksum: str
    product_text_schema_version: str
    provider: str
    model: str
    route_id: str
    dimension: int
    vector_dtype: str
    normalized: bool
    document_count: int
    artifact_format: str
    built_at: str
    artifact_checksums: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("index_id", "catalog_checksum", "ordered_id_checksum", "product_text_schema_version", "provider", "model", "route_id", "vector_dtype", "artifact_format", "built_at"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        _positive_integer(self.catalog_row_count, "catalog_row_count")
        _positive_integer(self.dimension, "dimension")
        _positive_integer(self.document_count, "document_count")
        if self.document_count != self.catalog_row_count:
            raise ValueError("document_count must match catalog_row_count")
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be a bool")
        checksums = _mapping(self.artifact_checksums, "artifact_checksums")
        if not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in checksums.items()):
            raise ValueError("artifact_checksums must contain non-empty string pairs")
        object.__setattr__(self, "artifact_checksums", checksums)
