"""Active-state query construction.

Retrieval consumes a structured plan built from validated state, never a
concatenation of raw conversation turns. That is the whole point: by the time a
plan is built, an override has already retracted what it replaced, so the plan
describes what the customer wants *now* rather than everything they ever said.

DG-02 lives here. The supplied profile is carried in `profile_bias`, visibly
separate from dialogue terms and filters, and it decays as explicit constraints
accumulate. At weight `0.0` it must be provably inert — see
`test_query_builder.py`, which asserts that a plan built with weight `0` is
identical to one built from a session that never had a profile at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tikitaka.contracts.domain import (
    Constraint,
    ProfileBias,
    SearchPlan,
)
from tikitaka.state.session import SessionState

# Deterministic term order, so a fixed state always yields a byte-identical
# plan. Category leads because it is the strongest retrieval signal.
ATTRIBUTE_ORDER = (
    "category",
    "style",
    "material",
    "color",
    "size",
    "feature",
    "use_case",
    "brand",
    "budget",
    "other",
)

# Attributes whose normalized value is numeric and belongs in a filter rather
# than in the free-text query.
NUMERIC_ATTRIBUTES = frozenset({"budget"})

PROFILE_TAG_KEY = "preference_tags"


@dataclass(frozen=True)
class QueryBuilderConfig:
    """Everything the P6 ablations need to vary without touching code."""

    profile_weight: float = 0.0
    profile_decay: float = 0.6
    max_profile_terms: int = 4
    include_revalidation_terms: bool = True
    route_policy: str = "auto"
    embedding_route_id: str | None = None
    index_id: str | None = None


class ActiveQueryBuilder:
    """Builds a `SearchPlan` from validated state. No model, no network."""

    def __init__(self, config: QueryBuilderConfig | None = None) -> None:
        self.config = config or QueryBuilderConfig()

    def build(self, state: SessionState) -> SearchPlan:
        active = tuple(state.active_constraints)
        revalidation = tuple(state.revalidation_constraints)

        includes = tuple(c for c in active if str(c.polarity) == "include")
        excludes = tuple(c for c in active if str(c.polarity) == "exclude")

        must = self._terms(c for c in includes if str(c.strength) == "hard")
        should = self._terms(c for c in includes if str(c.strength) == "soft")
        if self.config.include_revalidation_terms:
            # A flagged constraint is a maybe, so it may inform ranking but must
            # never harden into a filter until the customer reconfirms it.
            should = should + self._terms(revalidation)

        profile = self._profile_bias(state, len(includes))
        plan_terms = tuple(self._ordered_terms(includes))
        text_query = " ".join(plan_terms + profile.terms).strip()

        return SearchPlan(
            text_query=text_query,
            must_terms=must,
            should_terms=should,
            exclude_terms=self._terms(excludes),
            filters=self._filters(includes),
            attribute_values=self._attribute_values(includes),
            mode=str(state.mode),
            intent_version=state.intent_version,
            revalidation_flags=frozenset(str(c.attribute) for c in revalidation),
            no_preference=frozenset(str(item) for item in state.no_preference),
            profile_bias=profile,
            route_policy=self.config.route_policy,
            embedding_route_id=self.config.embedding_route_id,
            index_id=self.config.index_id,
        )

    # ---- internals ------------------------------------------------------

    def _terms(self, constraints: Iterable[Constraint]) -> tuple[str, ...]:
        """Text terms only. Numeric attributes are expressed as filters."""

        seen: list[str] = []
        for constraint in self._ordered(constraints):
            if str(constraint.attribute) in NUMERIC_ATTRIBUTES:
                continue
            term = str(constraint.normalized_value)
            if term and term not in seen:
                seen.append(term)
        return tuple(seen)

    def _ordered_terms(self, constraints: Iterable[Constraint]) -> list[str]:
        terms: list[str] = []
        for constraint in self._ordered(constraints):
            if str(constraint.attribute) in NUMERIC_ATTRIBUTES:
                continue
            term = str(constraint.normalized_value)
            if term and term not in terms:
                terms.append(term)
        return terms

    def _ordered(self, constraints: Iterable[Constraint]) -> list[Constraint]:
        def key(constraint: Constraint) -> tuple[int, int, str]:
            attribute = str(constraint.attribute)
            rank = (
                ATTRIBUTE_ORDER.index(attribute)
                if attribute in ATTRIBUTE_ORDER
                else len(ATTRIBUTE_ORDER)
            )
            return (rank, constraint.source_turn, str(constraint.normalized_value))

        return sorted(constraints, key=key)

    def _filters(self, includes: Iterable[Constraint]) -> dict[str, object]:
        """Structured filters Person 2's adapter can enforce directly.

        Budget becomes an `lte` bound. Other hard constraints are passed with
        their values so retrieval need not re-parse the text query.
        """

        filters: dict[str, object] = {}
        for constraint in self._ordered(includes):
            attribute = str(constraint.attribute)
            if attribute == "budget":
                bound = constraint.normalized_value
                if isinstance(bound, (int, float)):
                    filters["budget"] = {
                        "max": float(bound),
                        "polarity": "include",
                        "strength": str(constraint.strength),
                    }
                continue
            if str(constraint.strength) != "hard":
                continue
            existing = filters.get(attribute)
            values = tuple(existing["values"]) if isinstance(existing, dict) else ()
            filters[attribute] = {
                "values": values + (constraint.normalized_value,),
                "polarity": "include",
                "strength": "hard",
            }
        return filters

    def _attribute_values(
        self,
        includes: Iterable[Constraint],
    ) -> dict[str, tuple[object, ...]]:
        values: dict[str, tuple[object, ...]] = {}
        for constraint in self._ordered(includes):
            attribute = str(constraint.attribute)
            current = values.get(attribute, ())
            if constraint.normalized_value not in current:
                values[attribute] = current + (constraint.normalized_value,)
        return values

    def _profile_bias(self, state: SessionState, explicit_count: int) -> ProfileBias:
        """DG-02: a soft, decaying, separable signal that explicit state beats.

        Weight `0.0` produces an empty bias, so a profile-enabled build and a
        profile-free build are indistinguishable. That equality is what makes
        the held-out ablation honest rather than asserted.
        """

        weight = self.config.profile_weight
        if weight <= 0.0:
            return ProfileBias()

        terms = self._profile_terms(state)
        if not terms:
            return ProfileBias()

        decayed = weight * (self.config.profile_decay ** explicit_count)
        decayed = max(0.0, min(1.0, round(decayed, 6)))
        if decayed == 0.0:
            return ProfileBias()
        return ProfileBias(terms=terms, weight=decayed)

    def _profile_terms(self, state: SessionState) -> tuple[str, ...]:
        raw = state.profile_seed.get(PROFILE_TAG_KEY)
        if not isinstance(raw, (list, tuple)):
            return ()
        terms: list[str] = []
        for item in raw:
            text = str(item).strip().casefold()
            if text and text not in terms:
                terms.append(text)
        return tuple(terms[: self.config.max_profile_terms])


__all__ = [
    "ATTRIBUTE_ORDER",
    "ActiveQueryBuilder",
    "QueryBuilderConfig",
]
