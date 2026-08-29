"""Primary API interpreter for `gpt-5.6-terra` at `xhigh` reasoning.

Everything provider-specific lives below the `Transport` boundary: base URL,
auth header, request shape, and which parameter carries the reasoning level.
This module knows only that it sends a prompt and receives text plus token
counts, which is what lets the whole adapter be built and tested before the
endpoint is settled.

Model output stays untrusted. It is parsed through the same `state.schema`
boundary as any other interpreter, so a hallucinated attribute is dropped here
exactly as it would be from a fake.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from tikitaka.contracts.domain import StateDelta, Usage
from tikitaka.models import usage as usage_module
from tikitaka.models.base import (
    CredentialMissing,
    MalformedModelOutput,
    ModelRoute,
    ModelUnavailable,
)
from tikitaka.state.schema import (
    SCHEMA_VERSION,
    STRUCTURED_OUTPUT_SCHEMA,
    ParseResult,
    parse,
)

PROMPT_VERSION = "intent-interpreter/1"

_ATTRIBUTE_LIST = (
    "category, material, color, size, style, brand, budget, feature, "
    "use_case, other"
)

_SYSTEM_INSTRUCTIONS = f"""\
You convert one shopping message into structured state operations.

Return ONLY a JSON object matching the supplied schema. No prose, no markdown.

Allowed attributes: {_ATTRIBUTE_LIST}. Never invent an attribute outside this
list; use `other` when nothing fits.

Operations:
- add: a new constraint the customer just stated.
- replace: a correction of a value you were already told; name old_value.
- remove: the customer withdrew a constraint without replacing it.
- exclude: a negative constraint, such as "not leather".
- no_preference: the customer said the attribute does not matter to them.
- reset: the customer restarted. Use scope "conversation" for a full restart
  and "intent" when they switched to a different kind of product.

Distinguish carefully:
- "I don't have a preference for X; please use your judgment" is no_preference.
- "I don't have an additional preference for X" is NOT no_preference. It means
  they have nothing further to add about X. Emit no operation for it.

Set inferred_mode to "buying" when a concrete requirement is on the table, and
"browsing" when the customer is still exploring. Set generality high when the
request is vague and low when it is already specific.
"""


@dataclass(frozen=True)
class TransportResponse:
    """Provider-neutral result of one completed call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


class Transport(Protocol):
    """The only provider-aware seam.

    Implementations raise `ModelTimeout`, `ModelUnavailable`, or `ModelRefused`
    rather than leaking provider exception types upward.
    """

    def send(
        self,
        prompt: str,
        schema: Mapping[str, object],
        timeout_s: float,
    ) -> TransportResponse: ...


class ResponseCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...


class InMemoryResponseCache:
    """Replay cache for tuning runs. Off in reported runs."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, value: str) -> None:
        self._entries[key] = value


@dataclass(frozen=True)
class ApiConfig:
    route: ModelRoute
    timeout_s: float = 30.0
    max_repairs: int = 1
    transport_attempts: int = 2
    backoff_base_s: float = 0.5
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    cost_currency: str = "USD"
    cache: ResponseCache | None = None


class ApiInterpreter:
    """Satisfies `IntentInterpreter` using the primary API route."""

    def __init__(
        self,
        transport: Transport,
        config: ApiConfig,
        *,
        credential: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not (credential or "").strip():
            # Fail at construction so a missing key cannot surface as a
            # mid-session degradation on turn 7 of a scored run.
            raise CredentialMissing(
                "no API credential was supplied from the environment",
                config.route,
            )
        # The credential is checked for presence and deliberately NOT retained.
        # The transport owns the auth header, so the secret never enters this
        # object's __dict__, where a vars() dump, a debugger, or a serialized
        # error report would render it.
        self._transport = transport
        self._config = config
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"ApiInterpreter(route={self._config.route.route_id!r})"

    # ---- IntentInterpreter ---------------------------------------------

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        prompt = build_prompt(message, state)
        records: list[Usage] = []

        cached = self._from_cache(prompt)
        if cached is not None:
            result = parse(cached)
            if not result.top_level_failure:
                return result.delta, self._cache_usage()

        text, usage = self._send(prompt)
        records.append(usage)
        result = parse(text)

        repairs = 0
        while self._needs_repair(result) and repairs < self._config.max_repairs:
            repairs += 1
            repair_text, repair_usage = self._send(
                _repair_prompt(prompt, text), repair=True
            )
            records.append(repair_usage)
            text, result = repair_text, parse(repair_text)

        total = usage_module.accumulate(records)
        if self._needs_repair(result):
            # Tokens were really spent, so the failure carries its usage and
            # the extractor merges it before degrading to the heuristic route.
            error = MalformedModelOutput(
                "structured output failed validation after repair",
                self._config.route,
            )
            error.usage = total  # type: ignore[attr-defined]
            raise error

        self._to_cache(prompt, text)
        return result.delta, total

    # ---- internals ------------------------------------------------------

    def _send(self, prompt: str, *, repair: bool = False) -> tuple[str, Usage]:
        attempts = max(1, self._config.transport_attempts)
        last: Exception | None = None
        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = self._transport.send(
                    prompt, STRUCTURED_OUTPUT_SCHEMA, self._config.timeout_s
                )
            except ModelUnavailable as error:
                # Transient. A timeout is not retried here; it is a route
                # change, and burning the turn budget on a slow provider costs
                # more than falling back.
                last = error
                if attempt + 1 < attempts:
                    self._sleep(self._config.backoff_base_s * (2 ** attempt))
                continue
            latency_ms = (time.monotonic() - started) * 1000.0
            return response.text, usage_module.for_route(
                self._config.route,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                reasoning_tokens=response.reasoning_tokens,
                latency_ms=latency_ms,
                calls=1,
                repairs=1 if repair else 0,
                prompt_cost_per_1k=self._config.prompt_cost_per_1k,
                completion_cost_per_1k=self._config.completion_cost_per_1k,
                cost_currency=self._config.cost_currency,
            )
        raise last if last is not None else ModelUnavailable(
            "transport produced no response", self._config.route
        )

    def _needs_repair(self, result: ParseResult) -> bool:
        return result.top_level_failure

    def _cache_key(self, prompt: str) -> str:
        material = f"{self._config.route.route_id}\n{SCHEMA_VERSION}\n{prompt}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _from_cache(self, prompt: str) -> str | None:
        if self._config.cache is None:
            return None
        return self._config.cache.get(self._cache_key(prompt))

    def _to_cache(self, prompt: str, text: str) -> None:
        if self._config.cache is not None:
            self._config.cache.put(self._cache_key(prompt), text)

    def _cache_usage(self) -> Usage:
        return usage_module.for_route(
            self._config.route,
            cost_currency=self._config.cost_currency,
            cache_hit=True,
        )


def build_prompt(message: str, state: object) -> str:
    """Pure function of `(message, state, PROMPT_VERSION)` so calls replay.

    The model is given validated state, never the raw transcript, and never
    catalog products. Summarizing the conversation is the reducer's job and it
    has already happened; the shortlist token budget belongs to the reranker.
    """

    lines = [
        _SYSTEM_INSTRUCTIONS,
        f"prompt_version: {PROMPT_VERSION}",
        f"schema_version: {SCHEMA_VERSION}",
        "",
        "CURRENT STATE",
        f"mode: {getattr(state, 'mode', 'unknown')}",
        f"intent_version: {getattr(state, 'intent_version', 1)}",
        f"turn: {getattr(state, 'turn', 0)}",
    ]

    active = getattr(state, "active_constraints", ())
    if active:
        lines.append("active constraints:")
        for constraint in active:
            lines.append(
                f"  - {constraint.attribute}={constraint.normalized_value!r} "
                f"({constraint.polarity}, {constraint.strength})"
            )
    else:
        lines.append("active constraints: none")

    revalidation = getattr(state, "revalidation_constraints", ())
    if revalidation:
        lines.append("needs revalidation:")
        for constraint in revalidation:
            lines.append(f"  - {constraint.attribute}={constraint.normalized_value!r}")

    no_preference = sorted(str(item) for item in getattr(state, "no_preference", ()))
    if no_preference:
        lines.append(f"no preference: {', '.join(no_preference)}")

    lines += ["", "CUSTOMER MESSAGE", message or "", "", "JSON:"]
    return "\n".join(lines)


def _repair_prompt(original: str, bad_output: str) -> str:
    return (
        original
        + "\n\nYour previous reply was not valid against the schema:\n"
        + bad_output[:500]
        + "\n\nReturn only the corrected JSON object."
    )


def credential_from_env(variable: str, environ: Mapping[str, str]) -> str:
    """Read a credential by name. The value is never logged or returned home."""

    value = environ.get(variable, "")
    if not value.strip():
        raise CredentialMissing(f"environment variable {variable} is unset")
    return value


__all__ = [
    "PROMPT_VERSION",
    "ApiConfig",
    "ApiInterpreter",
    "InMemoryResponseCache",
    "ResponseCache",
    "Transport",
    "TransportResponse",
    "build_prompt",
    "credential_from_env",
]
