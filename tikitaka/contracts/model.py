"""Provider-neutral structural protocols used across workstreams."""

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


EmbeddingVector = tuple[float, ...]
EmbeddingBatch = tuple[EmbeddingVector, ...]


@runtime_checkable
class SessionStateView(Protocol):
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


@runtime_checkable
class IntentInterpreter(Protocol):
    def interpret(self, message: str, state: SessionStateView) -> tuple[StateDelta, Usage]: ...


@runtime_checkable
class QueryBuilder(Protocol):
    def build(self, state: SessionStateView) -> SearchPlan: ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]: ...


@runtime_checkable
class DecisionPolicy(Protocol):
    def choose(self, state: SessionStateView, candidates: Sequence[Candidate], turn: int) -> TurnDecision: ...


@runtime_checkable
class Reranker(Protocol):
    def rank(self, state: SessionStateView, candidates: Sequence[Candidate], top_k: int) -> tuple[list[str], Usage]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def route_id(self) -> str: ...
    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...
    def embed_query(self, text: str) -> EmbeddingVector: ...
