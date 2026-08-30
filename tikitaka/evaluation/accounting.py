"""Reconcile participant telemetry with an isolated provider billing snapshot.

This module is evaluation-only. It cannot discover why a provider bill differs
from locally reported usage, but it makes the boundary explicit: first prove
that every local component event was aggregated, then compare that total with a
provider snapshot taken over the same controlled window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from tikitaka.contracts import Usage
from tikitaka.models.usage import accumulate


@dataclass(frozen=True)
class ProviderUsageSnapshot:
    """Provider totals for one isolated run window, never a shared billing day."""

    prompt_tokens: int
    completion_tokens: int
    billed_cost: float | None = None
    cost_currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.billed_cost is not None:
            if (
                isinstance(self.billed_cost, bool)
                or not isinstance(self.billed_cost, (int, float))
                or not math.isfinite(float(self.billed_cost))
                or self.billed_cost < 0
            ):
                raise ValueError("billed_cost must be a non-negative finite number")
            object.__setattr__(self, "billed_cost", float(self.billed_cost))
        if (
            not isinstance(self.cost_currency, str)
            or len(self.cost_currency) != 3
            or not self.cost_currency.isalpha()
        ):
            raise ValueError("cost_currency must be a three-letter ISO-4217 code")
        object.__setattr__(self, "cost_currency", self.cost_currency.upper())


def _component_totals(events: tuple[tuple[str, Usage], ...]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Usage]] = {}
    for component, usage in events:
        if not isinstance(component, str) or not component.strip():
            raise ValueError("usage component must be a non-empty string")
        if not isinstance(usage, Usage):
            raise TypeError("usage event must contain canonical Usage")
        grouped.setdefault(component, []).append(usage)
    result: dict[str, dict[str, object]] = {}
    for component, records in sorted(grouped.items()):
        total = accumulate(records)
        result[component] = {
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "reasoning_tokens": total.reasoning_tokens,
            "calls": total.calls,
            "repairs": total.repairs,
            "estimated_cost": total.estimated_cost,
        }
    return result


def reconcile_usage(
    events: Iterable[tuple[str, Usage]],
    provider: ProviderUsageSnapshot | None = None,
    *,
    cost_tolerance: float = 1e-9,
) -> dict[str, object]:
    """Return deterministic local totals and any provider-side discrepancy.

    A provider comparison is meaningful only when its snapshot covers exactly
    the same isolated calls as ``events``. Shared-day billing totals must be
    reported as non-reconcilable rather than forced into this function.
    """

    if (
        isinstance(cost_tolerance, bool)
        or not isinstance(cost_tolerance, (int, float))
        or not math.isfinite(float(cost_tolerance))
        or cost_tolerance < 0
    ):
        raise ValueError("cost_tolerance must be a non-negative finite number")
    normalized = tuple(events)
    for item in normalized:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("usage events must be (component, Usage) pairs")
    components = _component_totals(normalized)
    total = accumulate(usage for _, usage in normalized)
    local = {
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "reasoning_tokens": total.reasoning_tokens,
        "calls": total.calls,
        "repairs": total.repairs,
        "estimated_cost": total.estimated_cost,
        "cost_currency": total.cost_currency,
    }
    result: dict[str, object] = {
        "status": "provider_snapshot_required",
        "local": local,
        "by_component": components,
        "provider": None,
        "delta_provider_minus_local": None,
    }
    if provider is None:
        return result
    if total.estimated_cost is not None and total.cost_currency != provider.cost_currency:
        raise ValueError("local and provider cost currencies do not match")
    prompt_delta = provider.prompt_tokens - total.prompt_tokens
    completion_delta = provider.completion_tokens - total.completion_tokens
    cost_delta = (
        None
        if provider.billed_cost is None or total.estimated_cost is None
        else provider.billed_cost - total.estimated_cost
    )
    token_deltas = (prompt_delta, completion_delta)
    cost_matches = cost_delta is None or abs(cost_delta) <= float(cost_tolerance)
    if all(delta == 0 for delta in token_deltas) and cost_matches:
        status = "matched"
    elif all(delta >= 0 for delta in token_deltas) and (
        any(delta > 0 for delta in token_deltas)
        or (cost_delta is not None and cost_delta > float(cost_tolerance))
    ):
        status = "local_underreported"
    elif all(delta <= 0 for delta in token_deltas) and (
        any(delta < 0 for delta in token_deltas)
        or (cost_delta is not None and cost_delta < -float(cost_tolerance))
    ):
        status = "local_overreported"
    else:
        status = "mixed_mismatch"
    result.update(
        status=status,
        provider={
            "prompt_tokens": provider.prompt_tokens,
            "completion_tokens": provider.completion_tokens,
            "billed_cost": provider.billed_cost,
            "cost_currency": provider.cost_currency,
        },
        delta_provider_minus_local={
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "cost": cost_delta,
        },
    )
    return result


__all__ = ["ProviderUsageSnapshot", "reconcile_usage"]
