"""Dependency-injected, label-free orchestration for one shopping turn."""

from __future__ import annotations

from threading import RLock
from typing import Generic, Protocol, Sequence, TypeVar

from tikitaka.contracts import (
    Attribute,
    Candidate,
    DecisionPolicy,
    IntentInterpreter,
    QueryBuilder,
    Reranker,
    Retriever,
    SessionStateView,
    StateDelta,
    TurnAction,
    TurnDecision,
    Usage,
)
from tikitaka.orchestration.sessions import SessionRegistry


StateT = TypeVar("StateT", bound=SessionStateView)
MAX_TURNS = 10
MAX_RECOMMENDATIONS = 10
SAFE_MESSAGE = "I couldn't complete that search safely. Please try again."


class StateReducer(Protocol[StateT]):
    def apply(self, state: StateT, delta: StateDelta, turn: int) -> StateT: ...

    def record_decision(
        self,
        state: StateT,
        decision: TurnDecision,
        shown_product_ids: Sequence[str],
    ) -> StateT: ...


class ShoppingAgent(Generic[StateT]):
    """Coordinate trusted components while validating every boundary."""

    def __init__(
        self,
        *,
        sessions: SessionRegistry[StateT],
        reducer: StateReducer[StateT],
        interpreter: IntentInterpreter,
        query_builder: QueryBuilder,
        retriever: Retriever,
        decision_policy: DecisionPolicy,
        reranker: Reranker,
        catalog_ids: set[str] | frozenset[str],
        candidate_limit: int = 100,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        self.sessions = sessions
        self._reducer = reducer
        self._interpreter = interpreter
        self._query_builder = query_builder
        self._retriever = retriever
        self._decision_policy = decision_policy
        self._reranker = reranker
        self._catalog_ids = frozenset(catalog_ids)
        self._candidate_limit = candidate_limit
        # Injected components may own non-thread-safe model clients or SQLite
        # connections. Keep each turn atomic across their shared pipeline.
        self._pipeline_lock = RLock()

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._pipeline_lock:
            self.sessions.reset(session_id, user_profile)

    def close(self) -> None:
        """Release optional resources owned by injected runtime components."""

        with self._pipeline_lock:
            close = getattr(self._retriever, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "ShoppingAgent[StateT]":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        with self._pipeline_lock:
            return self._respond_locked(session_id, user_message, turn, top_k)

    def _respond_locked(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if not self._valid_request(session_id, user_message, turn, top_k):
            return self._safe_response()
        state = self.sessions.get(session_id)
        if state is None:
            return self._safe_response()
        usage_offset = len(self.sessions.usage_events(session_id))

        state = self._interpret_and_apply(session_id, state, user_message, turn)
        candidates = self._retrieve(state)
        decision = self._choose(state, candidates, turn)
        if turn == MAX_TURNS and decision.action is TurnAction.CLARIFY:
            decision = self._recommend_fallback_decision()

        if decision.action is TurnAction.CLARIFY:
            shown: list[str] = []
            response = {
                "message": self._clarification_message(decision.ask_attribute),
                "ask_attribute": str(decision.ask_attribute),
                "recommendations": [],
            }
        else:
            ranked = self._rank(session_id, state, candidates, top_k)
            shown = self._normalize_ids(ranked, candidates, top_k)
            response = {
                "message": "Here are the closest matches I found.",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": item} for item in shown],
            }

        try:
            updated = self._reducer.record_decision(state, decision, shown)
            self.sessions.replace(session_id, updated)
        except Exception:
            # A history failure must not make the official payload invalid.
            pass

        prompt_tokens, completion_tokens = self._official_usage(session_id, usage_offset)
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return response

    @staticmethod
    def _valid_request(session_id: object, message: object, turn: object, top_k: object) -> bool:
        return (
            isinstance(session_id, str)
            and bool(session_id.strip())
            and isinstance(message, str)
            and isinstance(turn, int)
            and not isinstance(turn, bool)
            and 1 <= turn <= MAX_TURNS
            and isinstance(top_k, int)
            and not isinstance(top_k, bool)
            and top_k == MAX_RECOMMENDATIONS
        )

    def _interpret_and_apply(
        self,
        session_id: str,
        state: StateT,
        message: str,
        turn: int,
    ) -> StateT:
        try:
            result = self._interpreter.interpret(message, state)
            if not isinstance(result, tuple) or len(result) != 2:
                return state
            delta, usage = result
            if isinstance(usage, Usage):
                self.sessions.record_usage(session_id, "interpreter", usage)
            if not isinstance(delta, StateDelta):
                return state
            updated = self._reducer.apply(state, delta, turn)
            self.sessions.replace(session_id, updated)
            return updated
        except Exception:
            return state

    def _retrieve(self, state: StateT) -> list[Candidate]:
        try:
            plan = self._query_builder.build(state)
            raw = self._retriever.search(plan, self._candidate_limit)
        except Exception:
            return []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return []
        result: list[Candidate] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Candidate):
                continue
            if item.parent_asin not in self._catalog_ids or item.parent_asin in seen:
                continue
            seen.add(item.parent_asin)
            result.append(item)
        return result

    def _choose(
        self,
        state: StateT,
        candidates: Sequence[Candidate],
        turn: int,
    ) -> TurnDecision:
        try:
            decision = self._decision_policy.choose(state, candidates, turn)
            return decision if isinstance(decision, TurnDecision) else self._recommend_fallback_decision()
        except Exception:
            return self._recommend_fallback_decision()

    def _rank(
        self,
        session_id: str,
        state: StateT,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> Sequence[object]:
        fallback = [candidate.parent_asin for candidate in candidates]
        try:
            result = self._reranker.rank(state, candidates, top_k)
            if not isinstance(result, tuple) or len(result) != 2:
                return fallback
            ranked, usage = result
            if isinstance(usage, Usage):
                self.sessions.record_usage(session_id, "reranker", usage)
            if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
                return fallback
            return ranked
        except Exception:
            return fallback

    def _normalize_ids(
        self,
        ranked: Sequence[object],
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> list[str]:
        shortlist = {candidate.parent_asin for candidate in candidates}
        result: list[str] = []
        seen: set[str] = set()
        for item in ranked:
            parent_asin = item if isinstance(item, str) else ""
            if (
                not parent_asin
                or parent_asin in seen
                or parent_asin not in shortlist
                or parent_asin not in self._catalog_ids
            ):
                continue
            seen.add(parent_asin)
            result.append(parent_asin)
            if len(result) >= min(top_k, MAX_RECOMMENDATIONS):
                break
        for candidate in candidates:
            if candidate.parent_asin not in seen:
                seen.add(candidate.parent_asin)
                result.append(candidate.parent_asin)
            if len(result) >= min(top_k, MAX_RECOMMENDATIONS):
                break
        return result

    def _official_usage(self, session_id: str, offset: int) -> tuple[int, int]:
        events = self.sessions.usage_events(session_id)[offset:]
        return (
            sum(event.usage.prompt_tokens for event in events),
            sum(event.usage.completion_tokens for event in events),
        )

    @staticmethod
    def _recommend_fallback_decision() -> TurnDecision:
        from tikitaka.contracts import DecisionReasonCode

        return TurnDecision(
            action=TurnAction.RECOMMEND,
            ask_attribute=None,
            reason_code=DecisionReasonCode.COMPONENT_FALLBACK,
            reason="orchestration fallback",
            expected_information_gain=0.0,
        )

    @staticmethod
    def _clarification_message(attribute: Attribute | None) -> str:
        label = "one more preference" if attribute is None else str(attribute).replace("_", " ")
        return f"What {label} would help narrow this down?"

    @staticmethod
    def _safe_response() -> dict:
        return {
            "message": SAFE_MESSAGE,
            "ask_attribute": None,
            "recommendations": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
