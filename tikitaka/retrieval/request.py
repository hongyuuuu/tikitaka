"""Role 2 retrieval request and adapter for the frozen SearchPlan shape."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from .structured import ATTRIBUTE_NAMES
from .text import normalize_text


@dataclass(frozen=True, slots=True)
class RetrievalConstraint:
    """Retrieval-private normalized constraint, not a shared state contract."""

    attribute: str
    values: tuple[object, ...]
    polarity: str = "include"
    strength: str = "soft"
    operator: str = "eq"
    needs_revalidation: bool = False

    def __post_init__(self) -> None:
        if self.attribute not in ATTRIBUTE_NAMES:
            raise ValueError(f"unknown retrieval attribute: {self.attribute}")
        if not self.values:
            raise ValueError("retrieval constraint requires at least one value")
        if self.polarity not in {"include", "exclude"}:
            raise ValueError(f"invalid polarity: {self.polarity}")
        if self.strength not in {"hard", "soft"}:
            raise ValueError(f"invalid strength: {self.strength}")
        if self.operator not in {"eq", "lte", "lt", "gte", "gt"}:
            raise ValueError(f"invalid operator: {self.operator}")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    text_query: str
    must_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    constraints: tuple[RetrievalConstraint, ...] = ()
    mode: str = "unknown"
    intent_version: int = 1
    profile_terms: tuple[str, ...] = ()
    profile_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"buying", "browsing", "unknown"}:
            raise ValueError(f"invalid retrieval mode: {self.mode}")
        if self.intent_version <= 0:
            raise ValueError("intent_version must be positive")
        if not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be within [0.0, 1.0]")


def _values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(item for item in value if normalize_text(item))
    if isinstance(value, list):
        return tuple(item for item in value if normalize_text(item))
    return (value,) if normalize_text(value) else ()


def _constraint_from_filter(
    attribute: str,
    raw_filter: object,
    fallback_values: tuple[object, ...],
    *,
    needs_revalidation: bool,
) -> RetrievalConstraint | None:
    polarity = "include"
    strength = "hard"
    operator = "lte" if attribute == "budget" else "eq"
    values = fallback_values
    if isinstance(raw_filter, Mapping):
        polarity = str(raw_filter.get("polarity", polarity)).casefold()
        strength = str(raw_filter.get("strength", strength)).casefold()
        operator = str(raw_filter.get("operator", operator)).casefold()
        for key, key_operator in (
            ("max", "lte"),
            ("lte", "lte"),
            ("upper_bound", "lte"),
            ("min", "gte"),
            ("gte", "gte"),
            ("value", operator),
            ("values", operator),
        ):
            if key in raw_filter:
                values = _values(raw_filter[key])
                operator = key_operator
                break
    elif raw_filter is not None:
        values = _values(raw_filter)
    if not values:
        return None
    if attribute == "budget":
        normalized: list[object] = []
        for value in values:
            if isinstance(value, Decimal):
                normalized.append(value)
            else:
                normalized.append(value)
        values = tuple(normalized)
    return RetrievalConstraint(
        attribute=attribute,
        values=values,
        polarity=polarity,
        strength="soft" if needs_revalidation else strength,
        operator=operator,
        needs_revalidation=needs_revalidation,
    )


def request_from_search_plan(plan: object) -> RetrievalRequest:
    """Adapt the frozen 0.1.0 SearchPlan structurally without redefining it."""

    required_fields = (
        "text_query",
        "must_terms",
        "should_terms",
        "exclude_terms",
        "filters",
        "attribute_values",
        "mode",
        "intent_version",
        "revalidation_flags",
        "profile_bias",
    )
    missing = tuple(name for name in required_fields if not hasattr(plan, name))
    if missing:
        raise TypeError("SearchPlan-compatible object is missing: " + ", ".join(missing))
    attribute_values = getattr(plan, "attribute_values")
    filters = getattr(plan, "filters")
    if not isinstance(attribute_values, Mapping) or not isinstance(filters, Mapping):
        raise TypeError("SearchPlan attribute_values and filters must be mappings")
    revalidation = frozenset(str(value) for value in getattr(plan, "revalidation_flags"))
    constraints: list[RetrievalConstraint] = []
    for attribute in ATTRIBUTE_NAMES:
        fallback_values = _values(attribute_values.get(attribute))
        if attribute in filters:
            constraint = _constraint_from_filter(
                attribute,
                filters[attribute],
                fallback_values,
                needs_revalidation=attribute in revalidation,
            )
        elif fallback_values:
            constraint = RetrievalConstraint(
                attribute=attribute,
                values=fallback_values,
                strength="soft",
                needs_revalidation=attribute in revalidation,
            )
        else:
            constraint = None
        if constraint is not None:
            constraints.append(constraint)
    profile_bias = getattr(plan, "profile_bias")
    return RetrievalRequest(
        text_query=str(getattr(plan, "text_query")),
        must_terms=tuple(str(value) for value in getattr(plan, "must_terms")),
        should_terms=tuple(str(value) for value in getattr(plan, "should_terms")),
        exclude_terms=tuple(str(value) for value in getattr(plan, "exclude_terms")),
        constraints=tuple(constraints),
        mode=str(getattr(plan, "mode")),
        intent_version=int(getattr(plan, "intent_version")),
        profile_terms=tuple(str(value) for value in getattr(profile_bias, "terms", ())),
        profile_weight=float(getattr(profile_bias, "weight", 0.0)),
    )
