"""Candidate-grounded deterministic clarification phrasing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from tikitaka.contracts import Usage
from tikitaka.ranking.constraints import known_values, normalized_value
from tikitaka.ranking.deterministic import UsageRecord

from .diagnostics import ALLOWED_ATTRIBUTES


TEMPLATES: Mapping[str, str] = {
    "category": "Which product category are you looking for{examples}?",
    "material": "Do you have a material preference{examples}?",
    "color": "Which color would suit you best{examples}?",
    "size": "What size do you need{examples}?",
    "style": "Which style do you prefer{examples}?",
    "brand": "Do you have a preferred brand{examples}?",
    "budget": "What budget range should I stay within{examples}?",
    "feature": "Which feature matters most{examples}?",
    "use_case": "What will you mainly use it for{examples}?",
    "other": "What other preference would help narrow this down?",
}


def clarification_message(
    attribute: str,
    candidates: Sequence[object],
    maximum_examples: int = 2,
) -> str:
    if attribute not in ALLOWED_ATTRIBUTES:
        raise ValueError("attribute must be allowed by the official contract")
    if attribute == "other":
        return TEMPLATES[attribute]
    values: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for value in known_values(candidate, attribute):
            display = str(value).strip()
            key = normalized_value(value)
            if not display or key in seen:
                continue
            seen.add(key)
            values.append(display)
            if len(values) >= maximum_examples:
                break
        if len(values) >= maximum_examples:
            break
    examples = ""
    if len(values) == 1:
        examples = f", such as {values[0]}"
    elif len(values) >= 2:
        examples = f", for example {values[0]} or {values[1]}"
    return TEMPLATES[attribute].format(examples=examples)


def recommendation_message(count: int) -> str:
    if count <= 0:
        return "I could not find a reliable match yet, so I am keeping the search broad."
    return "Here are the strongest matches for your current preferences."


@dataclass(frozen=True)
class LLMClarifierConfig:
    model: str = "gpt-5.6-terra"
    reasoning_level: str = "xhigh"
    prompt_version: str = "person3-clarify-v1"
    maximum_candidates: int = 12
    maximum_message_length: int = 240

    def __post_init__(self) -> None:
        if self.model != "gpt-5.6-terra":
            raise ValueError("the frozen architecture permits only gpt-5.6-terra")
        if self.reasoning_level != "xhigh":
            raise ValueError("the frozen architecture requires xhigh reasoning")
        if min(self.maximum_candidates, self.maximum_message_length) <= 0:
            raise ValueError("clarifier limits must be positive")


@dataclass(frozen=True)
class ClarificationRequest:
    model: str
    reasoning_level: str
    prompt_version: str
    ask_attribute: str
    deterministic_message: str
    candidate_examples: tuple[str, ...]


class ClarificationModel(Protocol):
    def clarify(self, request: ClarificationRequest) -> tuple[object, object]: ...


class LLMClarifier:
    """Polish a fixed structured question without allowing action changes."""

    def __init__(
        self,
        model: ClarificationModel,
        config: LLMClarifierConfig | None = None,
        usage_type: type = Usage,
    ) -> None:
        self.model = model
        self.config = config or LLMClarifierConfig()
        self.usage_type = usage_type

    def phrase(
        self,
        attribute: str,
        candidates: Sequence[object],
    ) -> tuple[str, UsageRecord]:
        fallback = clarification_message(attribute, candidates)
        examples: list[str] = []
        for candidate in candidates[: self.config.maximum_candidates]:
            for value in known_values(candidate, attribute):
                display = str(value).strip()
                if display and display not in examples:
                    examples.append(display)
        request = ClarificationRequest(
            model=self.config.model,
            reasoning_level=self.config.reasoning_level,
            prompt_version=self.config.prompt_version,
            ask_attribute=attribute,
            deterministic_message=fallback,
            candidate_examples=tuple(examples[:4]),
        )
        start = time.perf_counter()
        try:
            output, usage = self.model.clarify(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return fallback, self.usage_type(
                calls=1,
                latency_ms=elapsed_ms,
                model=self.config.model,
                reasoning_level=self.config.reasoning_level,
                route="clarification_fallback",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if (
            not isinstance(output, str)
            or not output.strip()
            or len(output.strip()) > self.config.maximum_message_length
            or output.count("?") != 1
        ):
            return fallback, _clarifier_usage(
                usage,
                elapsed_ms,
                self.config,
                "clarification_fallback",
                self.usage_type,
            )
        return output.strip(), _clarifier_usage(
            usage, elapsed_ms, self.config, "llm_clarifier", self.usage_type
        )


def _safe_nonnegative(value: object, integer: bool = False) -> float | int:
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return 0 if integer else 0.0
    if not math.isfinite(float(result)):
        return 0 if integer else 0.0
    return max(0, result)


def _clarifier_usage(
    usage: object,
    elapsed_ms: float,
    config: LLMClarifierConfig,
    route: str,
    usage_type: type,
) -> object:
    cache_hit = bool(getattr(usage, "cache_hit", False))
    calls = 0 if cache_hit else max(1, int(_safe_nonnegative(getattr(usage, "calls", 1), True)))
    repairs = min(calls, int(_safe_nonnegative(getattr(usage, "repairs", 0), True)))
    estimated_cost = getattr(usage, "estimated_cost", None)
    if estimated_cost is not None:
        estimated_cost = float(_safe_nonnegative(estimated_cost))
    return usage_type(
        prompt_tokens=0 if cache_hit else int(_safe_nonnegative(getattr(usage, "prompt_tokens", 0), True)),
        completion_tokens=0 if cache_hit else int(_safe_nonnegative(getattr(usage, "completion_tokens", 0), True)),
        reasoning_tokens=0 if cache_hit else int(_safe_nonnegative(getattr(usage, "reasoning_tokens", 0), True)),
        calls=calls,
        repairs=repairs,
        latency_ms=0.0 if cache_hit else max(
            elapsed_ms, float(_safe_nonnegative(getattr(usage, "latency_ms", 0.0)))
        ),
        provider=getattr(usage, "provider", None),
        model=getattr(usage, "model", None) or config.model,
        reasoning_level=getattr(usage, "reasoning_level", None) or config.reasoning_level,
        estimated_cost=estimated_cost,
        cost_currency=getattr(usage, "cost_currency", "USD"),
        route=route,
        cache_hit=cache_hit,
    )
