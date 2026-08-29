#!/usr/bin/env python3
"""Run a pinned local dense/hybrid artifact and print evidence-rich diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import load_dense_index
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.request import RetrievalConstraint, RetrievalRequest


def _load_embedder(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--embedder-factory", required=True)
    parser.add_argument("--route", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--mode", choices=("buying", "browsing", "unknown"), default="unknown")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--material", action="append", default=[])
    parser.add_argument("--max-price", type=Decimal)
    arguments = parser.parse_args()

    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    embedder = _load_embedder(arguments.embedder_factory)
    index = load_dense_index(
        arguments.artifact,
        catalog,
        embedding_route_id=str(getattr(embedder, "route_id", "")),
    )
    constraints: list[RetrievalConstraint] = []
    if arguments.category:
        constraints.append(RetrievalConstraint("category", tuple(arguments.category)))
    if arguments.material:
        constraints.append(RetrievalConstraint("material", tuple(arguments.material)))
    if arguments.max_price is not None:
        constraints.append(
            RetrievalConstraint("budget", (arguments.max_price,), strength="hard", operator="lte")
        )
    request = RetrievalRequest(
        text_query=arguments.query,
        constraints=tuple(constraints),
        mode=arguments.mode,
        route_policy=arguments.route,
        embedding_route_id=index.manifest.route_id,
        index_id=index.manifest.index_id,
    )
    with HybridRetriever(catalog, dense_index=index, query_embedder=embedder) as retriever:
        result = retriever.retrieve(request, limit=arguments.limit)
    payload = {
        "diagnostics": {
            "requested_route": result.diagnostics.requested_route,
            "executed_route": result.diagnostics.executed_route,
            "route_failures": result.diagnostics.route_failures,
            "sparse_candidates": result.diagnostics.sparse_candidates,
            "dense_candidates": result.diagnostics.dense_candidates,
            "fused_candidates": result.diagnostics.fused_candidates,
            "hard_filtered_candidates": result.diagnostics.hard_filtered_candidates,
            "returned_candidates": result.diagnostics.returned_candidates,
            "route_overlap": dict(result.diagnostics.route_overlap),
            "top_score_margin": result.diagnostics.top_score_margin,
            "top_score_concentration": result.diagnostics.top_score_concentration,
            "effective_candidate_count": result.diagnostics.effective_candidate_count,
            "missing_attribute_rates": dict(result.diagnostics.missing_attribute_rates),
            "constraint_outcome_counts": {
                key: dict(value)
                for key, value in result.diagnostics.constraint_outcome_counts.items()
            },
            "route_timings_ms": dict(result.diagnostics.route_timings_ms),
            "manifest_ids": dict(result.diagnostics.manifest_ids),
        },
        "hits": [
            {
                "rank": rank,
                "parent_asin": hit.parent_asin,
                "title": hit.product.title,
                "price": None if hit.product.price is None else str(hit.product.price),
                "sparse_rank": hit.sparse_rank,
                "dense_rank": hit.dense_rank,
                "structural_score": hit.structural_score,
                "fused_score": hit.fused_score,
                "constraint_outcomes": {
                    item.attribute: item.outcome for item in hit.constraint_evaluations
                },
                "supporting_snippets": hit.supporting_snippets,
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
