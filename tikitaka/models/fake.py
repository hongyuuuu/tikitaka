"""Deterministic interpreters: scripted, heuristic, and faulty.

`HeuristicInterpreter` is production code, not a stub. It is the network-free
degraded route required by M5, so it must stay correct even once the API
adapter is the default.

Its vocabularies deliberately mirror the local evaluator's constraint
classifier, because the strings it parses are the same strings that classifier
bucketed. See `docs/PERSON_1_BUILD_PLAN.md` section 2.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from tikitaka.contracts.domain import StateDelta, StateOperation, Usage
from tikitaka.models.base import (
    MalformedModelOutput,
    ModelRoute,
    ModelTimeout,
)
from tikitaka.state.schema import (
    SCHEMA_VERSION,
    is_attribute,
    make_delta,
    operation,
    parse,
)

HEURISTIC_ROUTE = ModelRoute(
    route_id="heuristic/local",
    provider="local",
    model="heuristic-v1",
    pinned=True,
)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)
COLORS = ("black", "white", "blue", "red", "pink", "green")
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")

_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")
_LOOKING_FOR_RE = re.compile(r"i'm looking for ([^.,]+)", re.I)
_KEY_REQUIREMENT_RE = re.compile(r"a key requirement is:\s*(.+?)\.?$", re.I)
_WHAT_MATTERS_RE = re.compile(r"what matters is:\s*(.+?)\.?$", re.I)
_NO_ADDITIONAL_RE = re.compile(
    r"i don't have an additional preference for (\w+)", re.I
)
_NO_PREFERENCE_RE = re.compile(
    r"i don't have a preference for (\w+); please use your judgment", re.I
)
_EXPLORING_RE = re.compile(r"still exploring", re.I)
_OVERRIDE_RE = re.compile(
    r"\b(actually|instead|on second thought|forget|ignore my earlier|"
    r"changed my mind|rather)\b",
    re.I,
)


def classify_constraint(value: str) -> str:
    """Bucket a disclosed constraint string into an official attribute."""

    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if "color" in lowered or any(word in lowered for word in COLORS):
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(word in lowered for word in USE_CASE_WORDS):
        return "use_case"
    return "feature"


def carries_no_new_constraint(message: str) -> bool:
    """True when a reply is a recognised answer that adds no search constraint.

    Both official no-information templates count: the Boundary "I don't have a
    preference for X" and the spent-question "I don't have an additional
    preference for X".

    This exists because "the heuristic produced no operations" is ambiguous on
    its own — it means either *understood, nothing to add* or *failed to parse*.
    A caller that treats the first as the second will inject the customer's
    negative sentence into the retrieval query as if it were a preference.
    """

    text = message or ""
    return (
        _NO_ADDITIONAL_RE.search(text) is not None
        or _NO_PREFERENCE_RE.search(text) is not None
    )


def detect_exhaustion(message: str) -> str | None:
    """Return the attribute the customer had nothing further to add for.

    Deliberately distinct from a no-preference Boundary answer: this one means
    the question was spent, not that the attribute is irrelevant forever.
    """

    match = _NO_ADDITIONAL_RE.search(message)
    if match is None:
        return None
    attribute = match.group(1).lower()
    return attribute if attribute != "preference" else "other"


class ScriptedInterpreter:
    """Replays fixture deltas by turn index, for orchestration tests."""

    def __init__(self, deltas: Sequence[StateDelta]) -> None:
        self._deltas = tuple(deltas)
        self._calls = 0

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        index = min(self._calls, len(self._deltas) - 1) if self._deltas else -1
        self._calls += 1
        if index < 0:
            return make_delta(), Usage()
        return self._deltas[index], Usage()


class HeuristicInterpreter:
    """Regex extraction over the simulator's templates. No model, no network."""

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        text = (message or "").strip()
        operations: list[StateOperation] = []
        mode = "unknown"
        generality = 0.5

        boundary = _NO_PREFERENCE_RE.search(text)
        if boundary is not None:
            operations.append(
                operation(
                    "no_preference",
                    attribute=_as_attribute(boundary.group(1)),
                )
            )
            return self._delta(operations, mode, generality), Usage()

        if detect_exhaustion(text) is not None:
            # Handled by the extractor, which owns the distinction between a
            # spent question and a real no-preference answer.
            return self._delta(operations, mode, generality), Usage()

        is_override = _OVERRIDE_RE.search(text) is not None

        category = _LOOKING_FOR_RE.search(text)
        if category is not None:
            operations.append(
                operation(
                    "add",
                    attribute="category",
                    new_value=category.group(1).strip(),
                    polarity="include",
                    strength="hard",
                    confidence=0.9,
                )
            )

        requirement = _KEY_REQUIREMENT_RE.search(text)
        if requirement is not None:
            mode = "buying"
            generality = 0.3
            operations.extend(
                _constraint_operations(requirement.group(1), strength="hard")
            )

        matters = _WHAT_MATTERS_RE.search(text)
        if matters is not None:
            mode = "buying"
            generality = 0.35
            operations.extend(_constraint_operations(matters.group(1), strength="soft"))

        if _EXPLORING_RE.search(text) is not None:
            mode = "browsing"
            generality = 0.9

        if is_override:
            remainder = _OVERRIDE_RE.sub(" ", text)
            operations.extend(
                _constraint_operations(remainder, strength="hard")
            )
            generality = 0.4

        return self._delta(operations, mode, generality), Usage()

    def _delta(
        self,
        operations: list[StateOperation],
        mode: str,
        generality: float,
    ) -> StateDelta:
        return make_delta(
            inferred_mode=mode,
            mode_confidence=0.6 if mode != "unknown" else 0.0,
            operations=tuple(operations),
            generality=generality,
        )


@dataclass
class FaultyInterpreter:
    """Configurable failure source for the fault matrix."""

    mode: str = "malformed"
    payload: object = None
    calls: int = field(default=0, init=False)

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        self.calls += 1
        if self.mode == "timeout":
            raise ModelTimeout("provider did not answer", HEURISTIC_ROUTE)
        if self.mode == "exception":
            raise RuntimeError("component blew up")
        if self.mode == "malformed":
            result = parse("not json at all {{")
            return result.delta, Usage()
        if self.mode == "unknown_operation":
            result = parse(
                '{"inferred_mode": "buying", "operations": ['
                '{"operation": "obliterate", "attribute": "color"},'
                '{"operation": "add", "attribute": "color", "new_value": "red",'
                ' "polarity": "include", "strength": "soft", "confidence": 0.8}]}'
            )
            return result.delta, Usage()
        if self.mode == "raw":
            result = parse(self.payload)
            return result.delta, Usage()
        raise MalformedModelOutput(f"unknown fault mode {self.mode!r}", HEURISTIC_ROUTE)


def _constraint_operations(
    blob: str,
    *,
    strength: str,
) -> list[StateOperation]:
    operations: list[StateOperation] = []
    for part in _split_values(blob):
        attribute = classify_constraint(part)
        operations.append(
            operation(
                "add",
                attribute=attribute,
                new_value=part,
                polarity="include",
                strength=strength,
                confidence=0.8,
            )
        )
    return operations


def _split_values(blob: str) -> list[str]:
    parts = [part.strip(" .;,") for part in re.split(r"[;]", blob)]
    return [part for part in parts if len(part) > 1]


def _as_attribute(word: str) -> str:
    lowered = word.strip().lower()
    return lowered if is_attribute(lowered) else "other"


__all__ = [
    "HEURISTIC_ROUTE",
    "carries_no_new_constraint",
    "FaultyInterpreter",
    "HeuristicInterpreter",
    "ScriptedInterpreter",
    "classify_constraint",
    "detect_exhaustion",
]
