"""Provider-neutral protocols and the read-only session view.

Transcribed from `docs/p0/CONTRACT_PROPOSAL.md` section 4. `SessionStateView`
is structural: it never imports or names Person 1's concrete state class, which
is what keeps `tikitaka/contracts/` free of a cycle with `tikitaka/state/`.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from tikitaka.contracts.domain import (
    Attribute,
    Candidate,
    Constraint,
    InferredMode,
    SearchPlan,
    StateDelta,
    TurnDecision,
    Usage,
)


@runtime_checkable
class SessionStateView(Protocol):
    """Read-only projection handed to Persons 2 and 3."""

    @property
    def session_id(self) -> str: ...

    @property
    def turn(self) -> int: ...

    @property
    def mode(self) -> InferredMode: ...

    @property
    def mode_confidence(self) -> float: ...

    @property
    def intent_version(self) -> int: ...

    @property
    def active_constraints(self) -> tuple[Constraint, ...]: ...

    @property
    def revalidation_constraints(self) -> tuple[Constraint, ...]: ...

    @property
    def no_preference(self) -> frozenset[Attribute]: ...

    @property
    def asked_attributes(self) -> frozenset[Attribute]: ...

    @property
    def shown_product_ids(self) -> frozenset[str]: ...

    @property
    def profile_seed(self) -> Mapping[str, object]: ...


class IntentInterpreter(Protocol):
    def interpret(
        self,
        message: str,
        state: SessionStateView,
    ) -> tuple[StateDelta, Usage]: ...


class QueryBuilder(Protocol):
    def build(self, state: SessionStateView) -> SearchPlan: ...


class Retriever(Protocol):
    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]: ...


class DecisionPolicy(Protocol):
    def choose(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        turn: int,
    ) -> TurnDecision: ...


class Reranker(Protocol):
    def rank(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]: ...


__all__ = [
    "DecisionPolicy",
    "IntentInterpreter",
    "QueryBuilder",
    "Reranker",
    "Retriever",
    "SessionStateView",
]
