"""Structured turn traces for the demo and for error analysis.

A trace records what the agent believed after a turn and what it cost. It
carries no evaluator labels, no hidden target, no catalog rows, and no
credential — it is written from participant-visible state only, so it is safe
to paste into a report or a submission appendix.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from tikitaka.contracts.domain import Usage
from tikitaka.state.session import SessionState

# Guard list: nothing derived from evaluator internals may appear in a trace.
FORBIDDEN_KEYS = frozenset(
    {"ground_truth", "scenario_type", "intent_card", "behavior",
     "category_bucket", "difficulty_bucket"}
)

# Defence in depth. Our own transport never echoes a credential, but a future
# provider error, a proxy, or a dependency's exception text might, and a trace
# is written to disk and pasted into reports.
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}=*"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|token)\b\s*[:=]\s*\S+"),
)
REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Strip anything shaped like a credential from free text."""

    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


@dataclass(frozen=True)
class TurnTrace:
    session_id: str
    turn: int
    intent_version: int
    message: str
    mode: str
    generality: float
    active_constraints: tuple[dict, ...] = ()
    revalidation: tuple[str, ...] = ()
    no_preference: tuple[str, ...] = ()
    asked_attributes: tuple[str, ...] = ()
    exhausted_attributes: tuple[str, ...] = ()
    query_summary: str = ""
    route_id: str = ""
    used_fallback: bool = False
    failure: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    repairs: int = 0
    latency_ms: float = 0.0
    estimated_cost: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def capture(
    state: SessionState,
    message: str,
    turn: int,
    *,
    usage: Usage | None = None,
    route_id: str = "",
    used_fallback: bool = False,
    failure: str = "",
) -> TurnTrace:
    """Snapshot participant-visible state after a turn has been reduced."""

    usage = usage or Usage()
    return TurnTrace(
        session_id=state.session_id,
        turn=turn,
        intent_version=state.intent_version,
        message=message,
        mode=str(state.mode),
        generality=state.generality,
        active_constraints=tuple(
            {
                "attribute": str(constraint.attribute),
                "value": str(constraint.normalized_value),
                "polarity": str(constraint.polarity),
                "strength": str(constraint.strength),
                "source_turn": constraint.source_turn,
            }
            for constraint in state.active_constraints
        ),
        revalidation=tuple(
            str(constraint.attribute) for constraint in state.revalidation_constraints
        ),
        no_preference=tuple(sorted(str(item) for item in state.no_preference)),
        asked_attributes=tuple(sorted(str(item) for item in state.asked_attributes)),
        exhausted_attributes=tuple(
            sorted(str(item) for item in state.exhausted_attributes)
        ),
        query_summary=state.active_query_summary,
        route_id=route_id,
        used_fallback=used_fallback,
        failure=redact(failure)[:200],
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        calls=usage.calls,
        repairs=usage.repairs,
        latency_ms=round(usage.latency_ms, 3),
        estimated_cost=usage.estimated_cost,
    )


def write_jsonl(path: str | Path, traces: Iterable[TurnTrace]) -> Path:
    """Write traces as JSON Lines. Returns the path written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
    return target


def summarize(traces: Sequence[TurnTrace]) -> dict:
    """Aggregate one session's cost and route behaviour for the report."""

    return {
        "turns": len(traces),
        "final_intent_version": traces[-1].intent_version if traces else 1,
        "prompt_tokens": sum(item.prompt_tokens for item in traces),
        "completion_tokens": sum(item.completion_tokens for item in traces),
        "reasoning_tokens": sum(item.reasoning_tokens for item in traces),
        "calls": sum(item.calls for item in traces),
        "repairs": sum(item.repairs for item in traces),
        "fallback_turns": sum(1 for item in traces if item.used_fallback),
        "estimated_cost": sum(item.estimated_cost or 0.0 for item in traces),
    }


__all__ = [
    "FORBIDDEN_KEYS",
    "REDACTED",
    "TurnTrace",
    "capture",
    "redact",
    "summarize",
    "write_jsonl",
]
