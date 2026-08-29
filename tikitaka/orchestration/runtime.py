"""Composition root for the deterministic, network-free owner integration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tikitaka.contracts import (
    Attribute,
    ConstraintPolarity,
    ConstraintStrength,
    OperationScope,
    StateDelta,
    StateOperation,
    StateOperationKind,
)
from tikitaka.decision import ResponsePolicy, ResponsePolicyConfig
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent
from tikitaka.ranking import DeterministicRanker, DeterministicRankerConfig
from tikitaka.retrieval import SparseStructuredRetriever, load_catalog
from tikitaka.retrieval.retriever import RetrievalConfig
from tikitaka.state.query_builder import ActiveQueryBuilder, QueryBuilderConfig
from tikitaka.state.reducer import StateReducer
from tikitaka.state.session import SessionState, new_session


_VISIBLE_OVERRIDE_RE = re.compile(
    r"\b(actually|instead|on second thought|ignore my earlier|changed my mind|rather)\b",
    re.IGNORECASE,
)
_VISIBLE_OVERRIDE_VALUE_RE = re.compile(
    r"what i need is:\s*(.+?)\.?$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class DeterministicRuntimeConfig:
    """Pinned local route settings used by the official degraded path."""

    candidate_limit: int = 100
    profile_weight: float = 0.0
    query_builder: QueryBuilderConfig | None = None
    retrieval: RetrievalConfig | None = None
    decision: ResponsePolicyConfig | None = None
    ranking: DeterministicRankerConfig | None = None

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be within [0.0, 1.0]")


class VisibleMessageInterpreter:
    """Preserve arbitrary visible queries when the heuristic extracts no fields.

    The Person 1 heuristic understands the official simulator templates. This
    adapter adds the raw visible message as a soft ``other`` constraint only
    when that interpreter found no operation, so free-form official inputs do
    not collapse to an empty retrieval query.
    """

    def __init__(self, interpreter: object | None = None) -> None:
        self._interpreter = interpreter or HeuristicInterpreter()

    def interpret(self, message: str, state: object):
        delta, usage = self._interpreter.interpret(message, state)
        delta = self._mark_explicit_replacements(message, state, delta)
        if delta.operations or not message.strip():
            return delta, usage
        operation = StateOperation(
            operation=StateOperationKind.ADD,
            attribute=Attribute.OTHER,
            old_value=None,
            new_value=message.strip(),
            scope=OperationScope.ATTRIBUTE,
            polarity=ConstraintPolarity.INCLUDE,
            strength=ConstraintStrength.SOFT,
            confidence=0.5,
        )
        return StateDelta(
            inferred_mode=delta.inferred_mode,
            mode_confidence=delta.mode_confidence,
            operations=(operation,),
            generality=delta.generality,
            rejected_operations=delta.rejected_operations,
            schema_version=delta.schema_version,
        ), usage

    @staticmethod
    def _mark_explicit_replacements(
        message: str,
        state: object,
        delta: StateDelta,
    ) -> StateDelta:
        if _VISIBLE_OVERRIDE_RE.search(message) is None:
            return delta
        constraints_for = getattr(state, "constraints_for", None)
        if not callable(constraints_for):
            return delta
        corrected = _VISIBLE_OVERRIDE_VALUE_RE.search(message.strip())
        corrected_value = corrected.group(1).strip(" .") if corrected is not None else None
        operations: list[StateOperation] = []
        changed = False
        for item in delta.operations:
            if item.operation is not StateOperationKind.ADD or item.attribute is None:
                operations.append(item)
                continue
            existing = tuple(constraints_for(str(item.attribute)))
            if item.attribute is Attribute.CATEGORY and not existing:
                operations.append(item)
                continue
            old_value = existing[-1].value if existing else "superseded visible preference"
            operations.append(StateOperation(
                operation=StateOperationKind.REPLACE,
                attribute=item.attribute,
                old_value=old_value,
                new_value=corrected_value or item.new_value,
                scope=OperationScope.ATTRIBUTE,
                polarity=item.polarity,
                strength=item.strength,
                confidence=item.confidence,
            ))
            changed = True
        if not changed:
            return delta
        return StateDelta(
            inferred_mode=delta.inferred_mode,
            mode_confidence=delta.mode_confidence,
            operations=tuple(operations),
            generality=delta.generality,
            rejected_operations=delta.rejected_operations,
            schema_version=delta.schema_version,
        )


def build_deterministic_agent(
    catalog_path: str | Path,
    config: DeterministicRuntimeConfig | None = None,
) -> ShoppingAgent[SessionState]:
    """Wire the Person 1/2/3 implementations into Person 4 orchestration."""

    runtime = config or DeterministicRuntimeConfig()
    catalog = load_catalog(catalog_path)
    retriever = SparseStructuredRetriever(
        catalog,
        retrieval_config=runtime.retrieval,
    )
    query_config = runtime.query_builder or QueryBuilderConfig(
        profile_weight=runtime.profile_weight,
        route_policy="sparse",
    )
    sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
    return ShoppingAgent(
        sessions=sessions,
        reducer=StateReducer(),
        interpreter=VisibleMessageInterpreter(),
        query_builder=ActiveQueryBuilder(query_config),
        retriever=retriever,
        decision_policy=ResponsePolicy(config=runtime.decision),
        reranker=DeterministicRanker(config=runtime.ranking),
        catalog_ids=catalog.ids,
        candidate_limit=runtime.candidate_limit,
    )


__all__ = [
    "DeterministicRuntimeConfig",
    "VisibleMessageInterpreter",
    "build_deterministic_agent",
]
