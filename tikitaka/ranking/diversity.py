"""Controlled greedy diversity for early vague recommendation turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .constraints import active_constraints, enum_value, known_values, normalized_value
from .deterministic import ScoredCandidate


@dataclass(frozen=True)
class DiversityConfig:
    strength: float = 0.12
    attributes: tuple[str, ...] = ("category", "brand", "style", "material")

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("diversity strength must be in [0, 1]")


def _signature(candidate: object, attributes: Sequence[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for attribute in attributes:
        for value in known_values(candidate, attribute):
            tokens.add(f"{attribute}:{normalized_value(value)}")
    return frozenset(tokens)


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def diversify(
    state: object,
    ranked: Sequence[ScoredCandidate],
    top_k: int,
    config: DiversityConfig | None = None,
) -> tuple[ScoredCandidate, ...]:
    """Greedily diversify while preserving relevance as the dominant signal."""

    policy = config or DiversityConfig()
    if top_k <= 0 or not ranked:
        return ()

    explicit_attributes = {
        enum_value(getattr(constraint, "attribute", ""))
        for constraint in active_constraints(state)
    }
    attributes = tuple(
        attribute for attribute in policy.attributes if attribute not in explicit_attributes
    )
    if not attributes or policy.strength == 0:
        return tuple(ranked[:top_k])

    remaining = list(enumerate(ranked))
    signatures = {
        index: _signature(item.candidate, attributes) for index, item in remaining
    }
    chosen: list[tuple[int, ScoredCandidate]] = []
    max_score = max(item.score for item in ranked)
    min_score = min(item.score for item in ranked)
    width = max_score - min_score

    while remaining and len(chosen) < top_k:
        def utility(entry: tuple[int, ScoredCandidate]) -> tuple[float, float, str]:
            index, item = entry
            relevance = 1.0 if width == 0 else (item.score - min_score) / width
            redundancy = max(
                (_similarity(signatures[index], signatures[old_index]) for old_index, _ in chosen),
                default=0.0,
            )
            value = (1.0 - policy.strength) * relevance - policy.strength * redundancy
            return -value, -item.score, item.parent_asin

        best = min(remaining, key=utility)
        remaining.remove(best)
        chosen.append(best)

    return tuple(item for _, item in chosen)
