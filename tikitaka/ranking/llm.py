"""Bounded semantic shortlist reranking with strict deterministic validation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .constraints import active_constraints, enum_value
from .deterministic import DeterministicRanker, UsageRecord


@dataclass(frozen=True)
class LLMRerankerConfig:
    max_candidates: int = 50
    model: str = "gpt-5.6-terra"
    reasoning_level: str = "xhigh"
    prompt_version: str = "person3-rerank-v1"

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.model != "gpt-5.6-terra":
            raise ValueError("the frozen architecture permits only gpt-5.6-terra")
        if self.reasoning_level != "xhigh":
            raise ValueError("the frozen architecture requires xhigh reasoning")


@dataclass(frozen=True)
class RerankRequest:
    model: str
    reasoning_level: str
    prompt_version: str
    intent_version: int
    mode: str
    constraints: tuple[Mapping[str, object], ...]
    candidates: tuple[Mapping[str, object], ...]


class ShortlistRankingModel(Protocol):
    def rerank(self, request: RerankRequest) -> tuple[object, object]: ...


def _constraint_payload(state: object) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    for constraint in active_constraints(state):
        result.append(
            {
                "attribute": enum_value(getattr(constraint, "attribute", "")),
                "value": getattr(constraint, "normalized_value", getattr(constraint, "value", None)),
                "polarity": enum_value(getattr(constraint, "polarity", "include")),
                "strength": enum_value(getattr(constraint, "strength", "soft")),
                "confidence": float(getattr(constraint, "confidence", 0.0)),
            }
        )
    return tuple(result)


def _candidate_payload(candidate: object) -> Mapping[str, object]:
    evidence = getattr(candidate, "product_evidence", None)
    snippets = tuple(getattr(evidence, "supporting_snippets", ()) or ())[:4]
    return {
        "parent_asin": str(getattr(candidate, "parent_asin")),
        "fused_score": float(getattr(candidate, "fused_score", 0.0)),
        "structural_score": float(getattr(candidate, "structural_score", 0.0)),
        "matched_fields": tuple(getattr(evidence, "matched_fields", ()) or ())[:12],
        "supporting_snippets": tuple(str(item)[:300] for item in snippets),
        "constraint_outcomes": {
            enum_value(key): enum_value(value)
            for key, value in (getattr(evidence, "constraint_outcomes", {}) or {}).items()
        },
        "attribute_values": {
            enum_value(key): tuple(values)[:8]
            for key, values in (getattr(evidence, "attribute_values", {}) or {}).items()
        },
    }


def _extract_ids(output: object) -> list[str]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return []
    if isinstance(output, Mapping):
        for key in ("parent_asins", "ranked_parent_asins", "recommendations", "ids"):
            if key in output:
                output = output[key]
                break
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        return []
    result: list[str] = []
    for item in output:
        if isinstance(item, Mapping):
            item = item.get("parent_asin")
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, number) if math.isfinite(number) else 0.0


def _usage_record(
    usage: object,
    elapsed_ms: float,
    config: LLMRerankerConfig,
    usage_type: type,
    route: str = "llm_reranker",
) -> object:
    cache_hit = bool(getattr(usage, "cache_hit", False))
    calls = 0 if cache_hit else max(1, _nonnegative_int(getattr(usage, "calls", 1)))
    repairs = min(calls, _nonnegative_int(getattr(usage, "repairs", 0)))
    estimated_cost = getattr(usage, "estimated_cost", None)
    if estimated_cost is not None:
        estimated_cost = _nonnegative_float(estimated_cost)
    return usage_type(
        prompt_tokens=0 if cache_hit else _nonnegative_int(getattr(usage, "prompt_tokens", 0)),
        completion_tokens=0 if cache_hit else _nonnegative_int(getattr(usage, "completion_tokens", 0)),
        reasoning_tokens=0 if cache_hit else _nonnegative_int(getattr(usage, "reasoning_tokens", 0)),
        calls=calls,
        repairs=repairs,
        latency_ms=0.0 if cache_hit else max(
            elapsed_ms, _nonnegative_float(getattr(usage, "latency_ms", 0.0))
        ),
        provider=getattr(usage, "provider", None),
        model=getattr(usage, "model", None) or config.model,
        reasoning_level=getattr(usage, "reasoning_level", None) or config.reasoning_level,
        estimated_cost=estimated_cost,
        cost_currency=getattr(usage, "cost_currency", "USD"),
        route=route,
        cache_hit=cache_hit,
    )


class LLMReranker:
    def __init__(
        self,
        model: ShortlistRankingModel,
        deterministic: DeterministicRanker | None = None,
        config: LLMRerankerConfig | None = None,
        usage_type: type = UsageRecord,
    ) -> None:
        self.model = model
        self.deterministic = deterministic or DeterministicRanker()
        self.config = config or LLMRerankerConfig()
        self.usage_type = usage_type

    def rank(
        self,
        state: object,
        candidates: Sequence[object],
        top_k: int,
    ) -> tuple[list[str], object]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        deterministic_scored = self.deterministic.select_candidates(
            state, candidates, len(candidates)
        )
        deterministic_ids = [item.parent_asin for item in deterministic_scored]
        if not deterministic_ids or top_k == 0:
            return deterministic_ids[:top_k], self.usage_type(
                route="deterministic_fallback"
            )

        bounded = deterministic_scored[: self.config.max_candidates]
        request = RerankRequest(
            model=self.config.model,
            reasoning_level=self.config.reasoning_level,
            prompt_version=self.config.prompt_version,
            intent_version=int(getattr(state, "intent_version", 1)),
            mode=enum_value(getattr(state, "mode", "unknown")),
            constraints=_constraint_payload(state),
            candidates=tuple(_candidate_payload(item.candidate) for item in bounded),
        )
        start = time.perf_counter()
        try:
            raw_output, raw_usage = self.model.rerank(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return deterministic_ids[:top_k], self.usage_type(
                calls=1,
                latency_ms=elapsed_ms,
                model=self.config.model,
                reasoning_level=self.config.reasoning_level,
                route="deterministic_fallback",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = _usage_record(raw_usage, elapsed_ms, self.config, self.usage_type)

        allowed = {item.parent_asin for item in bounded}
        seen: set[str] = set()
        validated: list[str] = []
        for parent_asin in _extract_ids(raw_output):
            if parent_asin not in allowed or parent_asin in seen:
                continue
            seen.add(parent_asin)
            validated.append(parent_asin)

        if not validated:
            return deterministic_ids[:top_k], _usage_record(
                raw_usage,
                elapsed_ms,
                self.config,
                self.usage_type,
                route="deterministic_fallback",
            )
        for parent_asin in deterministic_ids:
            if parent_asin not in seen:
                seen.add(parent_asin)
                validated.append(parent_asin)

        return validated[:top_k], usage
