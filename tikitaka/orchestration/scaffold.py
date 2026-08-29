"""Replaceable deterministic scaffold used until owner components integrate.

This module deliberately provides only the minimum local vertical slice. The
state, retrieval, and decision owners replace these injected components without
changing :class:`ShoppingAgent` or the official adapter.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from tikitaka.config import STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.contracts import (
    Attribute, Candidate, Constraint, ConstraintPolarity, ConstraintStatus,
    ConstraintStrength, DecisionReasonCode, InferredMode, OperationScope,
    ProductEvidence, ProfileBias, RoutePolicy, SearchPlan, StateDelta,
    StateOperation, StateOperationKind, TurnAction, TurnDecision, Usage,
)
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
_ZERO_USAGE = Usage()


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


@dataclass(frozen=True)
class ScaffoldState:
    session_id: str
    profile_seed: Mapping[str, object]
    turn: int = 0
    mode: InferredMode = InferredMode.UNKNOWN
    mode_confidence: float = 0.0
    intent_version: int = 1
    active_constraints: tuple[Constraint, ...] = ()
    revalidation_constraints: tuple[Constraint, ...] = ()
    no_preference: frozenset[Attribute] = frozenset()
    asked_attributes: frozenset[Attribute] = frozenset()
    shown_product_ids: frozenset[str] = frozenset()


class ScaffoldInterpreter:
    def interpret(self, message: str, state: ScaffoldState) -> tuple[StateDelta, Usage]:
        operation = StateOperation(
            operation=StateOperationKind.ADD,
            attribute=Attribute.OTHER,
            old_value=None,
            new_value=message,
            scope=OperationScope.ATTRIBUTE,
            polarity=ConstraintPolarity.INCLUDE,
            strength=ConstraintStrength.SOFT,
            confidence=1.0,
        )
        return StateDelta(
            inferred_mode=InferredMode.UNKNOWN,
            mode_confidence=0.0,
            operations=(operation,),
            generality=0.5,
            rejected_operations=0,
            schema_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
        ), _ZERO_USAGE


class ScaffoldReducer:
    """Minimal reducer for the temporary scaffold, not the Person 1 reducer."""

    def apply(self, state: ScaffoldState, delta: StateDelta, turn: int) -> ScaffoldState:
        constraints = list(state.active_constraints)
        intent_version = state.intent_version
        no_preference = set(state.no_preference)
        for operation in delta.operations:
            if operation.operation is StateOperationKind.RESET:
                constraints.clear()
                no_preference.clear()
                intent_version += 1
            elif operation.operation is StateOperationKind.NO_PREFERENCE and operation.attribute:
                no_preference.add(operation.attribute)
                constraints = [item for item in constraints if item.attribute is not operation.attribute]
            elif operation.operation in (StateOperationKind.REMOVE, StateOperationKind.REPLACE):
                constraints = [item for item in constraints if item.attribute is not operation.attribute]
            if operation.operation in (
                StateOperationKind.ADD,
                StateOperationKind.EXCLUDE,
                StateOperationKind.REPLACE,
            ):
                constraints.append(Constraint(
                    attribute=operation.attribute or Attribute.OTHER,
                    value=operation.new_value,
                    normalized_value=operation.new_value,
                    polarity=operation.polarity or ConstraintPolarity.INCLUDE,
                    strength=operation.strength or ConstraintStrength.SOFT,
                    source_turn=turn,
                    confidence=operation.confidence or 0.0,
                    intent_version=intent_version,
                    status=ConstraintStatus.ACTIVE,
                ))
        return replace(
            state,
            turn=turn,
            mode=delta.inferred_mode,
            mode_confidence=delta.mode_confidence,
            intent_version=intent_version,
            active_constraints=tuple(constraints),
            no_preference=frozenset(no_preference),
            shown_product_ids=frozenset() if intent_version != state.intent_version else state.shown_product_ids,
        )

    def record_decision(
        self, state: ScaffoldState, decision: TurnDecision, shown_product_ids: Sequence[str]
    ) -> ScaffoldState:
        asked = set(state.asked_attributes)
        if decision.ask_attribute is not None:
            asked.add(decision.ask_attribute)
        shown = set(state.shown_product_ids)
        shown.update(shown_product_ids)
        return replace(state, asked_attributes=frozenset(asked), shown_product_ids=frozenset(shown))


class ScaffoldQueryBuilder:
    def build(self, state: ScaffoldState) -> SearchPlan:
        query = " ".join(
            str(item.normalized_value)
            for item in state.active_constraints
            if item.status is ConstraintStatus.ACTIVE
        )
        return SearchPlan(
            text_query=query,
            must_terms=(), should_terms=(), exclude_terms=(), filters={}, attribute_values={},
            mode=state.mode, intent_version=state.intent_version,
            revalidation_flags=frozenset(), no_preference=state.no_preference,
            profile_bias=ProfileBias(), route_policy=RoutePolicy.SPARSE,
        )


class ScaffoldRetriever:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_ids: set[str] = set()
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.catalog_ids.add(parent_asin)
                rows.append((
                    parent_asin, _text(product.get("title")), _text(product.get("categories")),
                    _text(product.get("features")), _text(product.get("details")),
                    _text(product.get("store")), _text(product.get("description")),
                ))
                if len(rows) >= 1000:
                    self._connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
                    rows.clear()
        if rows:
            self._connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        self._connection.commit()

    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]:
        terms = [
            token.lower() for token in _TOKEN_RE.findall(plan.text_query)
            if len(token) > 1 and token.lower() not in _STOPWORDS
        ]
        expression = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
        if not expression:
            return []
        rows = self._connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
            "FROM products WHERE products MATCH ? ORDER BY 2 LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [
            Candidate(
                parent_asin=str(parent_asin),
                product_evidence=ProductEvidence(
                    matched_fields=("catalog_text",), supporting_snippets=(),
                    constraint_outcomes={}, attribute_values={}, evidence_reliability={},
                    unknown_fields=(), route_details={"route": "scaffold_fts"},
                ),
                sparse_rank=rank, sparse_score=-float(score), dense_rank=None,
                dense_score=None, structural_score=0.0, fused_score=-float(score),
            )
            for rank, (parent_asin, score) in enumerate(rows, start=1)
        ]


class ScaffoldDecisionPolicy:
    def choose(self, state: ScaffoldState, candidates: Sequence[Candidate], turn: int) -> TurnDecision:
        return TurnDecision(
            action=TurnAction.RECOMMEND, ask_attribute=None,
            reason_code=DecisionReasonCode.RANKING_STABLE,
            reason="deterministic scaffold recommendation", expected_information_gain=0.0,
        )


class ScaffoldReranker:
    def rank(
        self, state: ScaffoldState, candidates: Sequence[Candidate], top_k: int
    ) -> tuple[list[str], Usage]:
        return [item.parent_asin for item in candidates[:top_k]], _ZERO_USAGE


def build_scaffold_agent(catalog_path: str | Path) -> ShoppingAgent[ScaffoldState]:
    retriever = ScaffoldRetriever(catalog_path)
    sessions = SessionRegistry(
        lambda session_id, profile: ScaffoldState(session_id=session_id, profile_seed=profile)
    )
    return ShoppingAgent(
        sessions=sessions, reducer=ScaffoldReducer(), interpreter=ScaffoldInterpreter(),
        query_builder=ScaffoldQueryBuilder(), retriever=retriever,
        decision_policy=ScaffoldDecisionPolicy(), reranker=ScaffoldReranker(),
        catalog_ids=retriever.catalog_ids,
    )
