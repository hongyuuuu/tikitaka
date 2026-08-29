"""Deterministic reciprocal-rank fusion for incomparable retrieval routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Iterable


@dataclass(frozen=True, slots=True)
class RRFConfig:
    k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    candidate_limit: int = 200

    def __post_init__(self) -> None:
        if self.k <= 0 or self.candidate_limit <= 0:
            raise ValueError("RRF k and candidate_limit must be positive")
        if not all(math.isfinite(value) for value in (self.sparse_weight, self.dense_weight)):
            raise ValueError("RRF route weights must be finite")
        if self.sparse_weight < 0 or self.dense_weight < 0:
            raise ValueError("RRF route weights must be non-negative")
        if self.sparse_weight == 0 and self.dense_weight == 0:
            raise ValueError("at least one RRF route weight must be positive")


@dataclass(frozen=True, slots=True)
class FusedRouteHit:
    parent_asin: str
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    route_score: float

    @property
    def best_rank(self) -> int:
        ranks = tuple(rank for rank in (self.sparse_rank, self.dense_rank) if rank is not None)
        return min(ranks) if ranks else 2**31 - 1


def _route_map(
    hits: Iterable[object],
    *,
    valid_ids: Collection[str] | None,
) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}
    for item in hits:
        parent_asin = getattr(item, "parent_asin", "")
        if not isinstance(parent_asin, str):
            raise ValueError("route hit ID must be a string")
        raw_rank = getattr(item, "rank", None)
        if isinstance(raw_rank, bool):
            raise ValueError("route hit rank must be an integer")
        try:
            rank = int(raw_rank)
            score = float(getattr(item, "score"))
        except (TypeError, ValueError) as error:
            raise ValueError("route hit must expose numeric rank and score") from error
        if not parent_asin or rank <= 0:
            raise ValueError("route hit must expose a non-empty ID and positive rank")
        if not math.isfinite(score):
            raise ValueError("route hit score must be finite")
        if valid_ids is not None and parent_asin not in valid_ids:
            continue
        previous = result.get(parent_asin)
        if previous is None or rank < previous[0] or (rank == previous[0] and score > previous[1]):
            result[parent_asin] = (rank, score)
    return result


def reciprocal_rank_fusion(
    sparse_hits: Iterable[object],
    dense_hits: Iterable[object],
    *,
    config: RRFConfig | None = None,
    valid_ids: Collection[str] | None = None,
) -> list[FusedRouteHit]:
    """Fuse route ranks while preserving route scores solely as evidence."""

    selected = config or RRFConfig()
    sparse = _route_map(sparse_hits, valid_ids=valid_ids)
    dense = _route_map(dense_hits, valid_ids=valid_ids)
    identifiers = set(sparse).union(dense)
    fused: list[FusedRouteHit] = []
    for parent_asin in identifiers:
        sparse_entry = sparse.get(parent_asin)
        dense_entry = dense.get(parent_asin)
        sparse_rank = None if sparse_entry is None else sparse_entry[0]
        sparse_score = None if sparse_entry is None else sparse_entry[1]
        dense_rank = None if dense_entry is None else dense_entry[0]
        dense_score = None if dense_entry is None else dense_entry[1]
        route_score = 0.0
        if sparse_rank is not None:
            route_score += selected.sparse_weight / (selected.k + sparse_rank)
        if dense_rank is not None:
            route_score += selected.dense_weight / (selected.k + dense_rank)
        fused.append(
            FusedRouteHit(
                parent_asin=parent_asin,
                sparse_rank=sparse_rank,
                sparse_score=sparse_score,
                dense_rank=dense_rank,
                dense_score=dense_score,
                route_score=route_score,
            )
        )
    fused.sort(
        key=lambda item: (
            -item.route_score,
            item.best_rank,
            item.sparse_rank if item.sparse_rank is not None else 2**31 - 1,
            item.dense_rank if item.dense_rank is not None else 2**31 - 1,
            item.parent_asin,
        )
    )
    return fused[: selected.candidate_limit]


def route_overlap(
    sparse_hits: Iterable[object],
    dense_hits: Iterable[object],
    *,
    depths: tuple[int, ...] = (10, 50, 100),
) -> dict[int, int]:
    sparse_ids = [str(getattr(item, "parent_asin", "")) for item in sparse_hits]
    dense_ids = [str(getattr(item, "parent_asin", "")) for item in dense_hits]
    return {
        depth: len(set(sparse_ids[:depth]).intersection(dense_ids[:depth]))
        for depth in depths
        if depth > 0
    }
