"""Composition roots for API-primary and deterministic owner integration."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from tikitaka.contracts import (
    Attribute,
    ConstraintPolarity,
    ConstraintStrength,
    OperationScope,
    StateDelta,
    StateOperation,
    StateOperationKind,
    Usage,
)
from tikitaka.decision import ResponsePolicy, ResponsePolicyConfig
from tikitaka.models.factory import GatewaySelection, gateway_from_env
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.models.selector import ModelSelector, RoutingInterpreter
from tikitaka.models.usage import merge
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent
from tikitaka.ranking import (
    DeterministicRanker,
    DeterministicRankerConfig,
    LLMReranker,
    LLMRerankerConfig,
    TextModelShortlistRanker,
)
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


@dataclass(frozen=True)
class RuntimeConfig(DeterministicRuntimeConfig):
    """Automatic primary route with an explicit deterministic contingency."""

    allow_degraded: bool = True
    enable_llm_reranker: bool = True
    llm_reranker: LLMRerankerConfig | None = None
    selector: ModelSelector | None = None


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


class ResilientInterpreter:
    """Run one selected route and degrade locally on any call failure."""

    def __init__(
        self,
        primary: object,
        route_id: str,
        fallback: object | None = None,
    ) -> None:
        self._primary = primary
        self._route_id = route_id
        self._fallback = fallback

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        try:
            delta, usage = self._primary.interpret(message, state)
            if isinstance(usage, Usage) and usage.route is None:
                usage = replace(usage, route=self._route_id)
            return delta, usage
        except Exception as error:
            if self._fallback is None:
                raise
            spent = getattr(error, "usage", None)
            spent = spent if isinstance(spent, Usage) else Usage()
            delta, fallback_usage = self._fallback.interpret(message, state)
            fallback_usage = (
                fallback_usage if isinstance(fallback_usage, Usage) else Usage()
            )
            usage = merge(spent, fallback_usage)
            return delta, replace(
                usage,
                route=f"{self._route_id}:fallback",
            )


def build_deterministic_agent(
    catalog_path: str | Path,
    config: DeterministicRuntimeConfig | None = None,
    *,
    interpreter: object | None = None,
    retriever: object | None = None,
) -> ShoppingAgent[SessionState]:
    """Wire the Person 1/2/3 implementations into Person 4 orchestration.

    `interpreter` overrides the deterministic route. Left as None this builds
    the fully local agent, which is what the M5 network-free run needs.
    """

    runtime = config or DeterministicRuntimeConfig()
    return _build_agent(
        catalog_path,
        runtime,
        interpreter=interpreter or VisibleMessageInterpreter(),
        reranker=DeterministicRanker(config=runtime.ranking),
        retriever=retriever,
        route_id="heuristic/local",
        degraded=True,
    )


def build_agent(
    catalog_path: str | Path,
    config: RuntimeConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    model_selection: GatewaySelection | None = None,
    retriever: object | None = None,
) -> tuple[ShoppingAgent[SessionState], str]:
    """Build the automatic API-primary route used by ``starter.Agent``.

    Returns the agent and the route id actually selected, so a report can
    never claim the API route while the deterministic one did the work.
    """

    runtime = config or RuntimeConfig()
    selection = model_selection or gateway_from_env(
        environ,
        allow_degraded=runtime.allow_degraded,
    )
    fallback = VisibleMessageInterpreter(HeuristicInterpreter())
    # Route per turn rather than once per build. A degraded selector has no
    # generative route, so it always chooses `fallback` and the deterministic
    # path behaves exactly as it did before routing existed.
    selector = runtime.selector or ModelSelector(
        None if selection.degraded else selection.route
    )
    interpreter = RoutingInterpreter(
        selector,
        fallback if selection.degraded else selection.interpreter,
        fallback,
    )
    deterministic = DeterministicRanker(config=runtime.ranking)
    reranker: object = deterministic
    if (
        runtime.enable_llm_reranker
        and not selection.degraded
        and selection.text_model is None
    ):
        raise ValueError("primary LLM reranking requires a structured text model")
    if (
        runtime.enable_llm_reranker
        and not selection.degraded
        and selection.text_model is not None
    ):
        reranker = LLMReranker(
            TextModelShortlistRanker(selection.text_model, selection.route),
            deterministic=deterministic,
            config=runtime.llm_reranker,
        )
    agent = _build_agent(
        catalog_path,
        runtime,
        interpreter=interpreter,
        reranker=reranker,
        retriever=retriever,
        route_id=selection.route.route_id,
        degraded=selection.degraded,
    )
    return agent, selection.route.route_id


def _build_agent(
    catalog_path: str | Path,
    runtime: DeterministicRuntimeConfig,
    *,
    interpreter: object,
    reranker: object,
    retriever: object | None,
    route_id: str,
    degraded: bool,
) -> ShoppingAgent[SessionState]:
    catalog = load_catalog(catalog_path)
    selected_retriever = retriever or SparseStructuredRetriever(
        catalog, retrieval_config=runtime.retrieval
    )
    query_config = runtime.query_builder or QueryBuilderConfig(
        profile_weight=runtime.profile_weight,
        route_policy="sparse",
    )
    sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
    return ShoppingAgent(
        sessions=sessions,
        reducer=StateReducer(),
        interpreter=interpreter,
        query_builder=ActiveQueryBuilder(query_config),
        retriever=selected_retriever,
        decision_policy=ResponsePolicy(config=runtime.decision),
        reranker=reranker,
        catalog_ids=catalog.ids,
        candidate_limit=runtime.candidate_limit,
        runtime_route_id=route_id,
        degraded=degraded,
    )


__all__ = [
    "DeterministicRuntimeConfig",
    "ResilientInterpreter",
    "RuntimeConfig",
    "VisibleMessageInterpreter",
    "build_agent",
    "build_deterministic_agent",
]
