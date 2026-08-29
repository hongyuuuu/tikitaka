"""Concrete mutable `SessionState`, owned by Person 1.

Person 4 owns the registry that stores one instance per session and never
mutates fields directly; it calls the reducer methods in `reducer.py`. Persons 2
and 3 receive this object only through the structural `SessionStateView`.

Deviation from `ARCHITECTURE.md` section 5, recorded deliberately: that sketch
types `active_constraints` as a `dict`, while frozen contract `0.1.0` requires
the view to expose `tuple[Constraint, ...]`. Constraints are stored internally
keyed by attribute and exposed as the contract's tuple. The frozen contract is
the later and more specific authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from tikitaka.contracts.domain import Constraint

# Attributes that may hold several simultaneous values.
MULTI_VALUED: frozenset[str] = frozenset({"feature", "use_case"})
MAX_VALUES_PER_ATTRIBUTE = 4

# Dependency table for DG-03 clearing on a major category change.
UNIVERSAL_ATTRIBUTES: frozenset[str] = frozenset({"budget"})
CATEGORY_DEPENDENT_ATTRIBUTES: frozenset[str] = frozenset(
    {"size", "style", "material", "feature", "use_case"}
)
AMBIGUOUS_ATTRIBUTES: frozenset[str] = frozenset({"color", "brand"})


@dataclass
class SessionState:
    """One isolated conversation. Never shared between sessions."""

    session_id: str
    turn: int = 0
    mode: str = "unknown"
    mode_confidence: float = 0.0
    intent_version: int = 1
    generality: float = 0.0
    active_query_summary: str = ""

    _constraints: dict[str, list[Constraint]] = field(default_factory=dict)
    constraint_history: list[Constraint] = field(default_factory=list)
    _no_preference: set[str] = field(default_factory=set)
    _asked: set[str] = field(default_factory=set)
    _exhausted: set[str] = field(default_factory=set)
    shown_by_intent: dict[int, set[str]] = field(default_factory=dict)
    candidate_set: list[str] = field(default_factory=list)
    profile_seed: Mapping[str, object] = field(default_factory=dict)

    # ---- SessionStateView surface -------------------------------------

    @property
    def active_constraints(self) -> tuple[Constraint, ...]:
        return tuple(
            constraint
            for constraint in self._iter_constraints()
            if constraint.status == "active"
        )

    @property
    def revalidation_constraints(self) -> tuple[Constraint, ...]:
        return tuple(
            constraint
            for constraint in self._iter_constraints()
            if constraint.status == "needs_revalidation"
        )

    @property
    def no_preference(self) -> frozenset[str]:
        return frozenset(self._no_preference)

    @property
    def asked_attributes(self) -> frozenset[str]:
        return frozenset(self._asked)

    @property
    def exhausted_attributes(self) -> frozenset[str]:
        """Attributes the customer answered with nothing further to add.

        Distinct from `no_preference`: exhaustion means the question was spent
        and returned no new constraint, while no-preference is an explicit
        Boundary answer that permanently suppresses the attribute.
        """

        return frozenset(self._exhausted)

    @property
    def shown_product_ids(self) -> frozenset[str]:
        """Products already shown under the *current* intent version.

        A new intent version makes prior products eligible again, which is what
        lets Person 3 re-offer a product that now fits.
        """

        return frozenset(self.shown_by_intent.get(self.intent_version, set()))

    # ---- Read helpers --------------------------------------------------

    def constraints_for(self, attribute: str) -> tuple[Constraint, ...]:
        return tuple(
            constraint
            for constraint in self._constraints.get(str(attribute), ())
            if constraint.status == "active"
        )

    def has_active(self, attribute: str) -> bool:
        return bool(self.constraints_for(attribute))

    def is_askable(self, attribute: str) -> bool:
        """Whether spending a turn on this attribute can still pay out."""

        return (
            attribute not in self._no_preference
            and attribute not in self._exhausted
            and attribute not in self._asked
        )

    def _iter_constraints(self) -> Iterable[Constraint]:
        for values in self._constraints.values():
            for constraint in values:
                yield constraint

    # ---- Mutation, reducer-only ----------------------------------------

    def _add(self, constraint: Constraint) -> None:
        bucket = self._constraints.setdefault(str(constraint.attribute), [])
        bucket.append(constraint)
        if constraint.attribute not in MULTI_VALUED:
            return
        active = [item for item in bucket if item.status == "active"]
        overflow = len(active) - MAX_VALUES_PER_ATTRIBUTE
        if overflow > 0:
            for stale in active[:overflow]:
                self._retire(stale, "replaced")

    def _retire(self, constraint: Constraint, status: str) -> None:
        bucket = self._constraints.get(str(constraint.attribute))
        if bucket is None:
            return
        for index, item in enumerate(bucket):
            if item is constraint:
                retired = replace(item, status=status)  # type: ignore[arg-type]
                bucket[index] = retired
                self.constraint_history.append(retired)
                return

    def _mark_shown(self, product_ids: Iterable[str]) -> None:
        bucket = self.shown_by_intent.setdefault(self.intent_version, set())
        bucket.update(product_ids)


def new_session(session_id: str, user_profile: Mapping[str, object] | None) -> SessionState:
    """Build an isolated session with a defensive copy of the supplied profile.

    The copy matters: the evaluator hands the same profile mapping to every
    session it creates, so retaining the caller's object would let one session
    observe another's mutation.
    """

    return SessionState(
        session_id=session_id,
        profile_seed=dict(user_profile or {}),
    )


__all__ = [
    "AMBIGUOUS_ATTRIBUTES",
    "CATEGORY_DEPENDENT_ATTRIBUTES",
    "MAX_VALUES_PER_ATTRIBUTE",
    "MULTI_VALUED",
    "UNIVERSAL_ATTRIBUTES",
    "SessionState",
    "new_session",
]
