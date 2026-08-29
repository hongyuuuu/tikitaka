"""Strict structured-output schema and the untrusted-input parse boundary.

This module is the only place that clamps. Everything downstream of `parse`
receives values the frozen contract already considers trusted, which is why the
domain records reject out-of-range input instead of silently repairing it.

Failure policy, per contract section 3.3:

- an invalid operation is rejected individually and counted;
- valid sibling operations survive;
- a top-level-invalid response yields a deterministic empty delta.

The sibling rule matters for scoring, not just tidiness: an Intent Override turn
carries a correction plus an addition in one delta, so whole-delta rejection on
one bad operation would drop the intent-version bump and damage the 15 percent
Override slice specifically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from tikitaka.contracts.domain import (
    ATTRIBUTES,
    ContractViolation,
    StateDelta,
    StateOperation,
)

SCHEMA_VERSION = "state-delta/1"

MAX_OPERATIONS = 24
MAX_VALUE_LENGTH = 120

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_BUDGET_RE = re.compile(r"(\d+(?:\.\d+)?)")
_WHITESPACE_RE = re.compile(r"\s+")

STRUCTURED_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["inferred_mode", "mode_confidence", "generality", "operations"],
    "properties": {
        "inferred_mode": {"enum": ["buying", "browsing", "unknown"]},
        "mode_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "generality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "query_summary": {"type": "string"},
        "operations": {
            "type": "array",
            "maxItems": MAX_OPERATIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation"],
                "properties": {
                    "operation": {
                        "enum": [
                            "add", "remove", "replace",
                            "exclude", "no_preference", "reset",
                        ]
                    },
                    "attribute": {"enum": sorted(ATTRIBUTES) + [None]},
                    "old_value": {},
                    "new_value": {},
                    "scope": {"enum": ["attribute", "conversation", "intent"]},
                    "polarity": {"enum": ["include", "exclude", None]},
                    "strength": {"enum": ["hard", "soft", None]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class ParseResult:
    """A validated delta plus everything that went wrong producing it."""

    delta: StateDelta
    query_summary: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)
    top_level_failure: bool = False

    @property
    def ok(self) -> bool:
        return not self.top_level_failure and not self.errors


def clamp_unit(value: object, default: float = 0.0) -> float:
    """Clamp untrusted numerics into `[0.0, 1.0]`.

    A non-numeric value becomes the configured default rather than an
    exception; booleans are not numbers here, since `True` for a confidence is
    a model mistake rather than the float `1.0`.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if numeric != numeric:  # NaN
        return default
    return max(0.0, min(1.0, numeric))


def normalize_text(value: object) -> str:
    """Deterministic matching form: collapsed whitespace, case-folded, capped."""

    text = _WHITESPACE_RE.sub(" ", str(value)).strip()
    return text.casefold()[:MAX_VALUE_LENGTH]


def normalize_budget(value: object) -> float | None:
    """Parse a currency-free upper bound, or `None` when no number is present.

    A failed parse never invents a number. The caller downgrades the operation
    to the `other` attribute instead.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    match = _BUDGET_RE.search(str(value).replace(",", ""))
    if match is None:
        return None
    return float(match.group(1))


def normalize_value(attribute: str, value: object) -> tuple[str, object] | None:
    """Return `(attribute, normalized_value)` or `None` when unusable.

    The attribute can change: an unparseable budget becomes `other` so the
    stated preference survives as text rather than being dropped or guessed at.
    """

    if value is None:
        return None
    if attribute == "budget":
        bound = normalize_budget(value)
        if bound is None:
            text = normalize_text(value)
            return ("other", text) if text else None
        return ("budget", bound)
    text = normalize_text(value)
    return (attribute, text) if text else None


def extract_json(raw: object) -> dict | None:
    """Recover an object from a string, a dict, or prose wrapping JSON."""

    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
    return parsed if isinstance(parsed, dict) else None


def parse(raw: object, *, default_mode: str = "unknown") -> ParseResult:
    """Parse untrusted model output into a validated `StateDelta`."""

    payload = extract_json(raw)
    if payload is None:
        return ParseResult(
            delta=StateDelta(schema_version=SCHEMA_VERSION),
            errors=("model output was not a parseable JSON object",),
            top_level_failure=True,
        )

    errors: list[str] = []
    operations: list[StateOperation] = []
    rejected = 0

    raw_operations = payload.get("operations")
    if raw_operations is None:
        raw_operations = []
    elif not isinstance(raw_operations, list):
        errors.append("operations was not a list")
        raw_operations = []

    if len(raw_operations) > MAX_OPERATIONS:
        errors.append(
            f"operation flood truncated from {len(raw_operations)} to {MAX_OPERATIONS}"
        )
        raw_operations = raw_operations[:MAX_OPERATIONS]

    for index, item in enumerate(raw_operations):
        operation, failure = _parse_operation(item)
        if operation is None:
            rejected += 1
            errors.append(f"operation {index}: {failure}")
            continue
        operations.append(operation)

    mode = payload.get("inferred_mode")
    if mode not in ("buying", "browsing", "unknown"):
        if mode is not None:
            errors.append(f"unknown inferred_mode {mode!r}")
        mode = default_mode

    summary = payload.get("query_summary")
    delta = StateDelta(
        inferred_mode=mode,  # type: ignore[arg-type]
        mode_confidence=clamp_unit(payload.get("mode_confidence")),
        operations=tuple(operations),
        generality=clamp_unit(payload.get("generality")),
        rejected_operations=rejected,
        schema_version=SCHEMA_VERSION,
    )
    return ParseResult(
        delta=delta,
        query_summary=normalize_text(summary) if isinstance(summary, str) else "",
        errors=tuple(errors),
    )


def empty_delta(mode: str = "unknown") -> StateDelta:
    """The deterministic fallback used when a model call fails outright."""

    return StateDelta(inferred_mode=mode, schema_version=SCHEMA_VERSION)  # type: ignore[arg-type]


def _parse_operation(item: object) -> tuple[StateOperation | None, str]:
    if not isinstance(item, dict):
        return None, "not an object"

    kind = item.get("operation")
    if not isinstance(kind, str):
        return None, "missing operation"
    kind = kind.strip().lower()

    attribute = item.get("attribute")
    if isinstance(attribute, str):
        attribute = attribute.strip().lower()
        if attribute not in ATTRIBUTES:
            return None, f"unknown attribute {attribute!r}"
    elif attribute is not None:
        return None, "attribute was not a string"

    new_value = item.get("new_value")
    old_value = item.get("old_value")

    if attribute is not None and new_value is not None:
        normalized = normalize_value(attribute, new_value)
        if normalized is None:
            return None, f"unusable value for {attribute!r}"
        attribute, new_value = normalized[0], new_value

    polarity = item.get("polarity")
    strength = item.get("strength")
    scope = item.get("scope")
    confidence = item.get("confidence")

    if kind in ("add", "replace", "exclude"):
        polarity = "exclude" if kind == "exclude" else _default(polarity, "include")
        strength = _default(strength, "soft")
        confidence = clamp_unit(confidence, default=0.5)
    elif kind == "remove":
        polarity = None
        strength = None
        new_value = None
        confidence = clamp_unit(confidence, default=0.5)
    elif kind == "no_preference":
        polarity = None
        strength = None
        new_value = None
        confidence = clamp_unit(confidence, default=1.0)
    elif kind == "reset":
        attribute = None
        polarity = None
        strength = None
        new_value = None
        old_value = None
        confidence = clamp_unit(confidence, default=1.0)
        if scope not in ("conversation", "intent"):
            scope = "conversation"
    else:
        return None, f"unknown operation {kind!r}"

    if scope not in ("attribute", "conversation", "intent"):
        scope = "attribute" if kind != "reset" else "conversation"

    try:
        return (
            StateOperation(
                operation=kind,  # type: ignore[arg-type]
                attribute=attribute,  # type: ignore[arg-type]
                old_value=old_value,
                new_value=new_value,
                scope=scope,  # type: ignore[arg-type]
                polarity=polarity,  # type: ignore[arg-type]
                strength=strength,  # type: ignore[arg-type]
                confidence=confidence,
            ),
            "",
        )
    except ContractViolation as error:
        return None, str(error)


def _default(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip().lower() in ("include", "exclude", "hard", "soft"):
        return value.strip().lower()
    return fallback


__all__ = [
    "MAX_OPERATIONS",
    "SCHEMA_VERSION",
    "STRUCTURED_OUTPUT_SCHEMA",
    "ParseResult",
    "clamp_unit",
    "empty_delta",
    "extract_json",
    "normalize_budget",
    "normalize_text",
    "normalize_value",
    "parse",
]
