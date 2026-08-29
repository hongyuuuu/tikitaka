"""Frozen `0.1.0` domain records.

Transcribed from `docs/p0/CONTRACT_PROPOSAL.md` sections 2 and 3. Serialized
values and validation semantics must match that document exactly.

Trusted internal constructors reject out-of-range values. Clamping belongs at
the untrusted-input boundary in `tikitaka/state/schema.py`, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

MAX_TURNS = 10
TOP_K = 10

Attribute = Literal[
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
]

ConstraintPolarity = Literal["include", "exclude"]
ConstraintStrength = Literal["hard", "soft"]
ConstraintStatus = Literal["active", "replaced", "retracted", "needs_revalidation"]
StateOperationKind = Literal[
    "add", "remove", "replace", "exclude", "no_preference", "reset",
]
OperationScope = Literal["attribute", "conversation", "intent"]
InferredMode = Literal["buying", "browsing", "unknown"]
TurnAction = Literal["clarify", "recommend"]
EvidenceOutcome = Literal["match", "contradiction", "unknown"]
RoutePolicy = Literal["auto", "sparse", "dense", "hybrid"]
DecisionReasonCode = Literal[
    "final_turn",
    "valuable_clarification",
    "low_question_value",
    "no_eligible_attribute",
    "ranking_stable",
    "insufficient_evidence",
    "component_fallback",
]

ATTRIBUTES: frozenset[str] = frozenset(
    {
        "category", "material", "color", "size", "style", "brand",
        "budget", "feature", "use_case", "other",
    }
)
POLARITIES: frozenset[str] = frozenset({"include", "exclude"})
STRENGTHS: frozenset[str] = frozenset({"hard", "soft"})
CONSTRAINT_STATUSES: frozenset[str] = frozenset(
    {"active", "replaced", "retracted", "needs_revalidation"}
)
OPERATION_KINDS: frozenset[str] = frozenset(
    {"add", "remove", "replace", "exclude", "no_preference", "reset"}
)
OPERATION_SCOPES: frozenset[str] = frozenset({"attribute", "conversation", "intent"})
INFERRED_MODES: frozenset[str] = frozenset({"buying", "browsing", "unknown"})
ROUTE_POLICIES: frozenset[str] = frozenset({"auto", "sparse", "dense", "hybrid"})
DECISION_REASON_CODES: frozenset[str] = frozenset(
    {
        "final_turn",
        "valuable_clarification",
        "low_question_value",
        "no_eligible_attribute",
        "ranking_stable",
        "insufficient_evidence",
        "component_fallback",
    }
)


class ContractViolation(ValueError):
    """A trusted constructor was given a value the frozen contract forbids."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def _require_unit_interval(value: float, name: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    _require(0.0 <= float(value) <= 1.0, f"{name} must be within [0.0, 1.0]")


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
    status: ConstraintStatus = "active"
    category_dependent: bool = False

    def __post_init__(self) -> None:
        _require(self.attribute in ATTRIBUTES, f"unknown attribute {self.attribute!r}")
        _require(self.polarity in POLARITIES, f"unknown polarity {self.polarity!r}")
        _require(self.strength in STRENGTHS, f"unknown strength {self.strength!r}")
        _require(
            self.status in CONSTRAINT_STATUSES, f"unknown status {self.status!r}"
        )
        _require(
            isinstance(self.source_turn, int) and 1 <= self.source_turn <= MAX_TURNS,
            "source_turn must be within the official 1-to-10 range",
        )
        _require(
            isinstance(self.intent_version, int) and self.intent_version >= 1,
            "intent_version must be positive",
        )
        _require_unit_interval(self.confidence, "confidence")

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class StateOperation:
    operation: StateOperationKind
    attribute: Attribute | None = None
    old_value: object | None = None
    new_value: object | None = None
    scope: OperationScope = "attribute"
    polarity: ConstraintPolarity | None = None
    strength: ConstraintStrength | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require(
            self.operation in OPERATION_KINDS, f"unknown operation {self.operation!r}"
        )
        _require(self.scope in OPERATION_SCOPES, f"unknown scope {self.scope!r}")
        if self.attribute is not None:
            _require(
                self.attribute in ATTRIBUTES, f"unknown attribute {self.attribute!r}"
            )
        if self.polarity is not None:
            _require(self.polarity in POLARITIES, f"unknown polarity {self.polarity!r}")
        if self.strength is not None:
            _require(self.strength in STRENGTHS, f"unknown strength {self.strength!r}")
        if self.confidence is not None:
            _require_unit_interval(self.confidence, "confidence")

        if self.operation in ("add", "exclude"):
            _require(self.attribute is not None, f"{self.operation} requires attribute")
            _require(self.new_value is not None, f"{self.operation} requires new_value")
            _require(self.polarity is not None, f"{self.operation} requires polarity")
            _require(self.strength is not None, f"{self.operation} requires strength")
            _require(
                self.confidence is not None, f"{self.operation} requires confidence"
            )
            if self.operation == "exclude":
                _require(
                    self.polarity == "exclude",
                    "exclude normalizes polarity to 'exclude'",
                )
        elif self.operation == "replace":
            _require(self.attribute is not None, "replace requires attribute")
            _require(self.new_value is not None, "replace requires new_value")
            _require(self.polarity is not None, "replace requires polarity")
            _require(self.strength is not None, "replace requires strength")
            _require(self.confidence is not None, "replace requires confidence")
        elif self.operation == "remove":
            _require(
                self.attribute is not None or self.old_value is not None,
                "remove requires an attribute or an unambiguous old value",
            )
            _require(
                self.new_value is None, "remove does not create a negative constraint"
            )
            _require(
                self.polarity is None, "remove does not create a negative constraint"
            )
        elif self.operation == "no_preference":
            _require(self.attribute is not None, "no_preference requires attribute")
            _require(self.new_value is None, "no_preference creates no constraint")
        elif self.operation == "reset":
            _require(
                self.scope in ("conversation", "intent"),
                "reset requires explicit 'conversation' or 'intent' scope",
            )
            _require(self.attribute is None, "reset does not name an attribute")


@dataclass(frozen=True)
class StateDelta:
    inferred_mode: InferredMode = "unknown"
    mode_confidence: float = 0.0
    operations: tuple[StateOperation, ...] = ()
    generality: float = 0.0
    rejected_operations: int = 0
    schema_version: str = ""

    def __post_init__(self) -> None:
        _require(
            self.inferred_mode in INFERRED_MODES,
            f"unknown inferred_mode {self.inferred_mode!r}",
        )
        _require_unit_interval(self.mode_confidence, "mode_confidence")
        _require_unit_interval(self.generality, "generality")
        _require(
            isinstance(self.rejected_operations, int)
            and self.rejected_operations >= 0,
            "rejected_operations must be a non-negative integer",
        )
        _require(isinstance(self.operations, tuple), "operations must be a tuple")


@dataclass(frozen=True)
class ProfileBias:
    terms: tuple[str, ...] = ()
    weight: float = 0.0

    def __post_init__(self) -> None:
        _require_unit_interval(self.weight, "profile weight")
        _require(isinstance(self.terms, tuple), "terms must be a tuple")

    @property
    def is_inert(self) -> bool:
        return self.weight == 0.0 or not self.terms


@dataclass(frozen=True)
class SearchPlan:
    text_query: str = ""
    must_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    filters: Mapping[str, object] = field(default_factory=dict)
    attribute_values: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    mode: InferredMode = "unknown"
    intent_version: int = 1
    revalidation_flags: frozenset[str] = frozenset()
    no_preference: frozenset[str] = frozenset()
    profile_bias: ProfileBias = field(default_factory=ProfileBias)
    route_policy: RoutePolicy = "auto"
    embedding_route_id: str | None = None
    index_id: str | None = None

    def __post_init__(self) -> None:
        _require(self.mode in INFERRED_MODES, f"unknown mode {self.mode!r}")
        _require(
            self.route_policy in ROUTE_POLICIES,
            f"unknown route_policy {self.route_policy!r}",
        )
        _require(
            isinstance(self.intent_version, int) and self.intent_version >= 1,
            "intent_version must be positive",
        )
        if self.route_policy == "dense":
            _require(
                self.embedding_route_id is not None and self.index_id is not None,
                "dense retrieval requires both embedding_route_id and index_id",
            )


@dataclass(frozen=True)
class ProductEvidence:
    matched_fields: tuple[str, ...] = ()
    supporting_snippets: tuple[str, ...] = ()
    constraint_outcomes: Mapping[str, EvidenceOutcome] = field(default_factory=dict)
    attribute_values: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    evidence_reliability: Mapping[str, float] = field(default_factory=dict)
    unknown_fields: tuple[str, ...] = ()
    route_details: Mapping[str, object] = field(default_factory=dict)
    profile_contribution: float = 0.0

    def __post_init__(self) -> None:
        for attribute, outcome in self.constraint_outcomes.items():
            _require(attribute in ATTRIBUTES, f"unknown attribute {attribute!r}")
            _require(
                outcome in ("match", "contradiction", "unknown"),
                f"unknown evidence outcome {outcome!r}",
            )
        for attribute, reliability in self.evidence_reliability.items():
            _require(attribute in ATTRIBUTES, f"unknown attribute {attribute!r}")
            _require_unit_interval(reliability, "evidence_reliability")
        _require_unit_interval(self.profile_contribution, "profile_contribution")


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    product_evidence: ProductEvidence = field(default_factory=ProductEvidence)
    sparse_rank: int | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    structural_score: float = 0.0
    fused_score: float = 0.0

    def __post_init__(self) -> None:
        _require(
            isinstance(self.parent_asin, str) and bool(self.parent_asin),
            "parent_asin must be a non-empty string",
        )
        for name in ("sparse_rank", "dense_rank"):
            rank = getattr(self, name)
            if rank is not None:
                _require(
                    isinstance(rank, int) and rank >= 1, f"{name} must be positive"
                )


@dataclass(frozen=True)
class TurnDecision:
    action: TurnAction
    ask_attribute: Attribute | None
    reason_code: DecisionReasonCode
    reason: str = ""
    expected_information_gain: float = 0.0

    def __post_init__(self) -> None:
        _require(
            self.action in ("clarify", "recommend"), f"unknown action {self.action!r}"
        )
        _require(
            self.reason_code in DECISION_REASON_CODES,
            f"unknown reason_code {self.reason_code!r}",
        )
        _require_unit_interval(
            self.expected_information_gain, "expected_information_gain"
        )
        if self.action == "clarify":
            _require(
                self.ask_attribute in ATTRIBUTES,
                "clarify requires one allowed ask_attribute",
            )
        else:
            _require(
                self.ask_attribute is None, "recommend requires ask_attribute is None"
            )


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
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "calls",
            "repairs",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{name} must be a non-negative integer",
            )
        _require(self.latency_ms >= 0.0, "latency_ms must be non-negative")
        _require(self.repairs <= self.calls, "repairs cannot exceed calls")
        if self.estimated_cost is not None:
            _require(self.estimated_cost >= 0.0, "estimated_cost must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


__all__ = [
    "MAX_TURNS",
    "TOP_K",
    "ATTRIBUTES",
    "Attribute",
    "Candidate",
    "Constraint",
    "ConstraintPolarity",
    "ConstraintStatus",
    "ConstraintStrength",
    "ContractViolation",
    "DecisionReasonCode",
    "EvidenceOutcome",
    "InferredMode",
    "OperationScope",
    "ProductEvidence",
    "ProfileBias",
    "RoutePolicy",
    "SearchPlan",
    "StateDelta",
    "StateOperation",
    "StateOperationKind",
    "TurnAction",
    "TurnDecision",
    "Usage",
]
