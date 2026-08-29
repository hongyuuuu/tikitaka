"""The validating state reducer. All state mutation lives here.

Operations are applied in a fixed order so a multi-operation delta is
reproducible: reset, replace, remove and exclude, add, then no-preference. The
order is a decision rather than an accident, because one turn can carry a
correction and an addition together — which is exactly what an Intent Override
turn does.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from tikitaka.contracts.domain import (
    Constraint,
    StateDelta,
    StateOperation,
    TurnDecision,
)
from tikitaka.state.schema import normalize_value
from tikitaka.state.session import (
    AMBIGUOUS_ATTRIBUTES,
    CATEGORY_DEPENDENT_ATTRIBUTES,
    MULTI_VALUED,
    SessionState,
)

_ORDER = {
    "reset": 0,
    "replace": 1,
    "remove": 2,
    "exclude": 3,
    "add": 4,
    "no_preference": 5,
}


class StateReducer:
    """Deterministic, validated mutation of one session's state."""

    def apply(
        self,
        state: SessionState,
        delta: StateDelta,
        turn: int,
    ) -> SessionState:
        state.turn = turn
        if delta.inferred_mode != "unknown":
            state.mode = delta.inferred_mode
            state.mode_confidence = delta.mode_confidence
        state.generality = delta.generality

        # A schema-level REPLACE represents an explicit customer correction.
        # Advance once so products shown for the superseded intent are eligible
        # again, without clearing unrelated constraints. Category changes own
        # their dependency-aware version transition below.
        has_non_category_replace = any(
            operation.operation == "replace" and operation.attribute != "category"
            for operation in delta.operations
        )
        has_category_change = any(
            _changes_category(state, operation)
            for operation in delta.operations
        )
        if has_non_category_replace and not has_category_change:
            self._new_intent_version(state)

        for operation in sorted(delta.operations, key=_sort_key):
            self._apply_operation(state, operation, turn)
        return state

    def record_decision(
        self,
        state: SessionState,
        decision: TurnDecision,
        shown_product_ids: Sequence[str] = (),
    ) -> SessionState:
        """Record the turn's outcome. Called by orchestration, owned here."""

        if decision.action == "clarify" and decision.ask_attribute is not None:
            state._asked.add(decision.ask_attribute)
        if shown_product_ids:
            state._mark_shown(shown_product_ids)
        return state

    def note_exhausted(self, state: SessionState, attribute: str) -> SessionState:
        """The customer had nothing further for an attribute we asked about.

        Not the same as no-preference: exhaustion is a spent question, while
        no-preference is an explicit Boundary answer.
        """

        state._exhausted.add(attribute)
        state._asked.add(attribute)
        return state

    # ---- operations -----------------------------------------------------

    def _apply_operation(
        self,
        state: SessionState,
        operation: StateOperation,
        turn: int,
    ) -> None:
        kind = operation.operation
        if kind == "reset":
            self._reset(state, operation.scope)
        elif kind == "replace":
            self._upsert(state, operation, turn, force_replace=True)
        elif kind == "remove":
            self._remove(state, operation)
        elif kind == "exclude":
            self._exclude(state, operation, turn)
        elif kind == "add":
            self._upsert(state, operation, turn, force_replace=False)
        elif kind == "no_preference":
            self._no_preference(state, operation)

    def _reset(self, state: SessionState, scope: str) -> None:
        """DG-03 restart paths.

        `conversation` is an explicit start-over: every conversation-derived
        constraint goes, and only the profile snapshot survives. `intent` is a
        softer pivot that keeps still-applicable universal constraints such as
        budget, so "actually, show me sweaters instead" does not silently
        discard a stated price ceiling.
        """

        if scope == "intent":
            self._category_change(state)
            return
        for constraint in state.active_constraints:
            state._retire(constraint, "retracted")
        for constraint in state.revalidation_constraints:
            state._retire(constraint, "retracted")
        self._new_intent_version(state)

    def _upsert(
        self,
        state: SessionState,
        operation: StateOperation,
        turn: int,
        *,
        force_replace: bool,
    ) -> None:
        attribute = operation.attribute
        if attribute is None or operation.new_value is None:
            return
        normalized = normalize_value(attribute, operation.new_value)
        if normalized is None:
            return
        attribute, normalized_value = normalized

        existing = state.constraints_for(attribute)

        # An add that repeats a known value refreshes provenance instead of
        # stacking a duplicate.
        for constraint in existing:
            if constraint.normalized_value == normalized_value:
                refreshed = replace(
                    constraint,
                    source_turn=turn,
                    confidence=max(constraint.confidence, operation.confidence or 0.0),
                )
                self._swap(state, constraint, refreshed)
                return

        if attribute == "category" and existing:
            # A different product type is a new intent, not another filter.
            self._category_change(state)

        if existing and (force_replace or attribute not in MULTI_VALUED):
            for constraint in existing:
                state._retire(constraint, "replaced")

        state._add(
            Constraint(
                attribute=attribute,  # type: ignore[arg-type]
                value=operation.new_value,
                normalized_value=normalized_value,
                polarity=operation.polarity or "include",  # type: ignore[arg-type]
                strength=operation.strength or "soft",  # type: ignore[arg-type]
                source_turn=turn,
                confidence=operation.confidence or 0.0,
                intent_version=state.intent_version,
                category_dependent=attribute in CATEGORY_DEPENDENT_ATTRIBUTES,
            )
        )

    def _remove(self, state: SessionState, operation: StateOperation) -> None:
        attribute = operation.attribute
        targets: tuple[Constraint, ...]
        if attribute is not None:
            targets = state.constraints_for(attribute)
        else:
            wanted = normalize_value("other", operation.old_value)
            if wanted is None:
                return
            targets = tuple(
                constraint
                for constraint in state.active_constraints
                if constraint.normalized_value == wanted[1]
            )
        for constraint in targets:
            state._retire(constraint, "retracted")

    def _exclude(
        self,
        state: SessionState,
        operation: StateOperation,
        turn: int,
    ) -> None:
        attribute = operation.attribute
        if attribute is None or operation.new_value is None:
            return
        normalized = normalize_value(attribute, operation.new_value)
        if normalized is None:
            return
        attribute, normalized_value = normalized

        # An exclusion beats a stale include of the same value; the include is
        # retracted rather than left to contradict the exclusion silently.
        for constraint in state.constraints_for(attribute):
            if (
                constraint.polarity == "include"
                and constraint.normalized_value == normalized_value
            ):
                state._retire(constraint, "retracted")

        state._add(
            Constraint(
                attribute=attribute,  # type: ignore[arg-type]
                value=operation.new_value,
                normalized_value=normalized_value,
                polarity="exclude",
                strength=operation.strength or "soft",  # type: ignore[arg-type]
                source_turn=turn,
                confidence=operation.confidence or 0.0,
                intent_version=state.intent_version,
                category_dependent=attribute in CATEGORY_DEPENDENT_ATTRIBUTES,
            )
        )

    def _no_preference(self, state: SessionState, operation: StateOperation) -> None:
        attribute = operation.attribute
        if attribute is None:
            return
        state._no_preference.add(attribute)
        state._asked.add(attribute)

    # ---- DG-03 ----------------------------------------------------------

    def _category_change(self, state: SessionState) -> None:
        """Dependency-aware clearing for a major category change.

        Universal constraints such as budget survive, constraints derived from
        or incompatible with the old category are retracted, and ambiguous
        survivors are flagged for revalidation rather than silently enforced or
        silently dropped.
        """

        for constraint in state.active_constraints:
            attribute = constraint.attribute
            if attribute == "category":
                continue
            if attribute in CATEGORY_DEPENDENT_ATTRIBUTES:
                state._retire(constraint, "retracted")
            elif attribute in AMBIGUOUS_ATTRIBUTES:
                self._swap(
                    state, constraint, replace(constraint, status="needs_revalidation")
                )
        self._new_intent_version(state)

    def _new_intent_version(self, state: SessionState) -> None:
        """Begin a new intent version and reopen the question budget.

        Previously shown products become eligible again because
        `shown_product_ids` is scoped to the current version, and suppressed
        attributes reopen because a no-preference answer was given about the
        old intent, not this one.
        """

        state.intent_version += 1
        state._no_preference.clear()
        state._asked.clear()
        state._exhausted.clear()

    def _swap(
        self,
        state: SessionState,
        old: Constraint,
        new: Constraint,
    ) -> None:
        bucket = state._constraints.get(str(old.attribute))
        if bucket is None:
            return
        for index, item in enumerate(bucket):
            if item is old:
                bucket[index] = new
                return


def _sort_key(operation: StateOperation) -> tuple[int, str]:
    return (_ORDER.get(str(operation.operation), 99), str(operation.attribute or ""))


def _changes_category(state: SessionState, operation: StateOperation) -> bool:
    if operation.attribute != "category" or operation.new_value is None:
        return False
    normalized = normalize_value("category", operation.new_value)
    if normalized is None:
        return False
    existing = state.constraints_for("category")
    return bool(existing) and all(
        constraint.normalized_value != normalized[1]
        for constraint in existing
    )


__all__ = ["StateReducer"]
