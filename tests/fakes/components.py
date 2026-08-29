"""Fake-first component implementations for contract and orchestration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tikitaka.config import STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.contracts import (
    Attribute,
    Candidate,
    Constraint,
    DecisionReasonCode,
    InferredMode,
    ProductEvidence,
    ProfileBias,
    RoutePolicy,
    SearchPlan,
    SessionStateView,
    StateDelta,
    TurnAction,
    TurnDecision,
    Usage,
)


FAKE_USAGE = Usage(
    prompt_tokens=7,
    completion_tokens=3,
    calls=1,
    latency_ms=1.0,
    provider="fake",
    model="scripted-v1",
    reasoning_level="none",
    estimated_cost=0.0,
    route="fake",
)


def empty_delta() -> StateDelta:
    return StateDelta(
        inferred_mode=InferredMode.UNKNOWN,
        mode_confidence=0.0,
        operations=(),
        generality=1.0,
        rejected_operations=0,
        schema_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class FakeSessionState:
    session_id: str = "fake-session"
    turn: int = 1
    mode: InferredMode = InferredMode.UNKNOWN
    mode_confidence: float = 0.0
    intent_version: int = 1
    active_constraints: tuple[Constraint, ...] = ()
    revalidation_constraints: tuple[Constraint, ...] = ()
    no_preference: frozenset[Attribute] = frozenset()
    asked_attributes: frozenset[Attribute] = frozenset()
    shown_product_ids: frozenset[str] = frozenset()
    profile_seed: Mapping[str, object] = field(default_factory=dict)


class ScriptedInterpreter:
    def __init__(
        self,
        script: Mapping[str, StateDelta] | None = None,
        default: StateDelta | None = None,
        usage: Usage = FAKE_USAGE,
        seed: int = 0,
    ) -> None:
        self._script = dict(script or {})
        self._default = default or empty_delta()
        self._usage = usage
        self.seed = seed

    def interpret(self, message: str, state: SessionStateView) -> tuple[StateDelta, Usage]:
        return self._script.get(message, self._default), self._usage


class FakeQueryBuilder:
    def __init__(self, plan: SearchPlan | None = None, seed: int = 0) -> None:
        self._plan = plan
        self.seed = seed

    def build(self, state: SessionStateView) -> SearchPlan:
        if self._plan is not None:
            return self._plan
        return SearchPlan(
            text_query="",
            must_terms=(),
            should_terms=(),
            exclude_terms=(),
            filters={},
            attribute_values={},
            mode=state.mode,
            intent_version=state.intent_version,
            revalidation_flags=frozenset(),
            no_preference=state.no_preference,
            profile_bias=ProfileBias(),
            route_policy=RoutePolicy.SPARSE,
        )


class DeterministicRetriever:
    def __init__(
        self,
        candidates: Sequence[Candidate],
        catalog_ids: set[str] | frozenset[str],
        seed: int = 0,
    ) -> None:
        invalid = {candidate.parent_asin for candidate in candidates} - set(catalog_ids)
        if invalid:
            raise ValueError(f"fake candidates are not catalog-valid: {sorted(invalid)!r}")
        self._candidates = tuple(sorted(candidates, key=Candidate.retrieval_sort_key))
        self.seed = seed

    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return list(self._candidates[:limit])


class ScriptedDecisionPolicy:
    def __init__(
        self,
        decisions: Mapping[int, TurnDecision] | None = None,
        default: TurnDecision | None = None,
        seed: int = 0,
    ) -> None:
        self._decisions = dict(decisions or {})
        self._default = default or TurnDecision(
            action=TurnAction.RECOMMEND,
            ask_attribute=None,
            reason_code=DecisionReasonCode.RANKING_STABLE,
            reason="deterministic fake recommendation",
            expected_information_gain=0.0,
        )
        self.seed = seed

    def choose(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        turn: int,
    ) -> TurnDecision:
        return self._decisions.get(turn, self._default)


class DeterministicReranker:
    """A bounded reranker that filters its script through shortlist identity."""

    def __init__(
        self,
        preferred_ids: Sequence[str] = (),
        usage: Usage = FAKE_USAGE,
        seed: int = 0,
    ) -> None:
        self._preferred_ids = tuple(dict.fromkeys(preferred_ids))
        self._usage = usage
        self.seed = seed

    def rank(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        shortlist = tuple(candidate.parent_asin for candidate in candidates)
        shortlist_ids = set(shortlist)
        ordered = [item for item in self._preferred_ids if item in shortlist_ids]
        seen = set(ordered)
        for item in shortlist:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered[:top_k], self._usage


class MalformedInterpreter:
    def interpret(self, message: str, state: SessionStateView) -> tuple[object, Usage]:
        return {"inferred_mode": "not-a-mode"}, FAKE_USAGE


class MalformedReranker:
    def rank(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]:
        return ["OUTSIDE_SHORTLIST", "OUTSIDE_SHORTLIST"], FAKE_USAGE


class RaisingInterpreter:
    def interpret(self, message: str, state: SessionStateView) -> tuple[StateDelta, Usage]:
        raise RuntimeError("scripted interpreter failure")


class RaisingRetriever:
    def search(self, plan: SearchPlan, limit: int) -> list[Candidate]:
        raise RuntimeError("scripted retriever failure")


class RaisingDecisionPolicy:
    def choose(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        turn: int,
    ) -> TurnDecision:
        raise RuntimeError("scripted decision failure")


class RaisingReranker:
    def rank(
        self,
        state: SessionStateView,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> tuple[list[str], Usage]:
        raise RuntimeError("scripted reranker failure")


def evidence(
    *,
    matched_fields: tuple[str, ...] = ("title",),
    unknown_fields: tuple[str, ...] = (),
) -> ProductEvidence:
    return ProductEvidence(
        matched_fields=matched_fields,
        supporting_snippets=("synthetic evidence",),
        constraint_outcomes={},
        attribute_values={},
        evidence_reliability={},
        unknown_fields=unknown_fields,
        route_details={"source": "fake"},
    )


def candidate(parent_asin: str, rank: int, score: float) -> Candidate:
    return Candidate(
        parent_asin=parent_asin,
        product_evidence=evidence(),
        sparse_rank=rank,
        sparse_score=score,
        dense_rank=None,
        dense_score=None,
        structural_score=0.0,
        fused_score=score,
    )
