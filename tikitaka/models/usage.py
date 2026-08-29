"""Usage accumulation and cost estimation.

Orchestration aggregates one record per component call. A repair retry is a
second provider call inside one logical `interpret()`, so an adapter returns a
single merged record whose `calls` and `repairs` counters make the retry
reconcilable without provider-specific types.
"""

from __future__ import annotations

from dataclasses import replace

from tikitaka.contracts.domain import Usage
from tikitaka.models.base import ModelRoute

EMPTY_USAGE = Usage()


def merge(left: Usage, right: Usage) -> Usage:
    """Combine two records from the same logical call.

    Identity fields come from the left record when present, so a merged repair
    keeps the original route attribution. `cache_hit` is true only when every
    merged part was served from cache.
    """

    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        calls=left.calls + right.calls,
        repairs=left.repairs + right.repairs,
        latency_ms=left.latency_ms + right.latency_ms,
        provider=left.provider or right.provider,
        model=left.model or right.model,
        reasoning_level=left.reasoning_level or right.reasoning_level,
        estimated_cost=_add_optional(left.estimated_cost, right.estimated_cost),
        cost_currency=left.cost_currency or right.cost_currency,
        route=left.route or right.route,
        cache_hit=left.cache_hit and right.cache_hit,
    )


def accumulate(records: object) -> Usage:
    """Sum an iterable of usage records into one."""

    total = EMPTY_USAGE
    for record in records:  # type: ignore[union-attr]
        total = merge(total, record)
    return total


def _add_optional(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return (left or 0.0) + (right or 0.0)


def for_route(
    route: ModelRoute,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    latency_ms: float = 0.0,
    calls: int = 1,
    repairs: int = 0,
    prompt_cost_per_1k: float = 0.0,
    completion_cost_per_1k: float = 0.0,
    cost_currency: str = "USD",
    cache_hit: bool = False,
) -> Usage:
    """Build one attributable record for a completed provider call.

    `completion_tokens` is the billed output total. `reasoning_tokens` is an
    informational *subset* of it, not an addition: the provider reports it
    under `completion_tokens_details`, verified against a live response. Adding
    the two would double-count every reasoning token, which at `xhigh` would
    inflate the reported cost badly.

    A cache hit adds no call, tokens, latency, or cost to the current run.
    """

    if cache_hit:
        return Usage(
            provider=route.provider,
            model=route.model,
            reasoning_level=route.reasoning_level,
            route=route.route_id,
            cost_currency=cost_currency,
            cache_hit=True,
        )

    estimated_cost = (
        prompt_tokens / 1000.0 * prompt_cost_per_1k
        + completion_tokens / 1000.0 * completion_cost_per_1k
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        calls=calls,
        repairs=repairs,
        latency_ms=latency_ms,
        provider=route.provider,
        model=route.model,
        reasoning_level=route.reasoning_level,
        estimated_cost=estimated_cost,
        cost_currency=cost_currency,
        route=route.route_id,
    )


def redacted(usage: Usage) -> Usage:
    """Drop provider identity for logs shared outside the model package."""

    return replace(usage, provider=None, model=None, route=None)


__all__ = ["EMPTY_USAGE", "accumulate", "for_route", "merge", "redacted"]
