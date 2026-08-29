#!/usr/bin/env python3
"""Inspect the deterministic sparse/structured route on a local catalog."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tikitaka.retrieval import (
    RetrievalConstraint,
    RetrievalRequest,
    SparseStructuredRetriever,
    load_catalog,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Current active-state search query, not a full transcript")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--mode", choices=("buying", "browsing", "unknown"), default="unknown")
    parser.add_argument("--intent-version", type=int, default=1)
    parser.add_argument("--must", action="append", default=[])
    parser.add_argument("--should", action="append", default=[])
    parser.add_argument("--exclude-term", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--material", action="append", default=[])
    parser.add_argument("--exclude-material", action="append", default=[])
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--use-case", action="append", default=[])
    parser.add_argument("--max-price", type=Decimal)
    return parser


def _constraints(arguments: argparse.Namespace) -> tuple[RetrievalConstraint, ...]:
    constraints: list[RetrievalConstraint] = []
    for attribute in ("category", "material", "feature", "use_case"):
        values = tuple(getattr(arguments, attribute))
        if values:
            constraints.append(RetrievalConstraint(attribute, values, strength="soft"))
    if arguments.exclude_material:
        constraints.append(
            RetrievalConstraint(
                "material",
                tuple(arguments.exclude_material),
                polarity="exclude",
                strength="hard",
            )
        )
    if arguments.max_price is not None:
        constraints.append(
            RetrievalConstraint("budget", (arguments.max_price,), strength="hard", operator="lte")
        )
    return tuple(constraints)


def main() -> int:
    arguments = _parser().parse_args()
    started = perf_counter()
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    loaded = perf_counter()
    with SparseStructuredRetriever(catalog) as retriever:
        indexed = perf_counter()
        result = retriever.retrieve(
            RetrievalRequest(
                text_query=arguments.query,
                must_terms=tuple(arguments.must),
                should_terms=tuple(arguments.should),
                exclude_terms=tuple(arguments.exclude_term),
                constraints=_constraints(arguments),
                mode=arguments.mode,
                intent_version=arguments.intent_version,
            ),
            limit=arguments.limit,
        )
        finished = perf_counter()
        payload = {
            "catalog": {
                "rows": len(catalog),
                "source_sha256": catalog.identity.source_sha256,
                "ordered_parent_asin_sha256": catalog.identity.ordered_parent_asin_sha256,
            },
            "timing_ms": {
                "catalog_load": round((loaded - started) * 1_000, 3),
                "index_build": round((indexed - loaded) * 1_000, 3),
                "search": round((finished - indexed) * 1_000, 3),
            },
            "diagnostics": {
                "route": result.diagnostics.route,
                "sparse_candidates": result.diagnostics.sparse_candidates,
                "hard_filtered_candidates": result.diagnostics.hard_filtered_candidates,
                "returned_candidates": result.diagnostics.returned_candidates,
                "intent_version": result.diagnostics.intent_version,
                "manifest": dict(result.diagnostics.sparse_manifest),
            },
            "hits": [
                {
                    "rank": rank,
                    "parent_asin": hit.parent_asin,
                    "title": hit.product.title,
                    "price": None if hit.product.price is None else str(hit.product.price),
                    "sparse_rank": hit.sparse_rank,
                    "sparse_score": hit.sparse_score,
                    "structural_score": hit.structural_score,
                    "fused_score": hit.fused_score,
                    "matched_fields": hit.matched_fields,
                    "constraint_outcomes": {
                        item.attribute: item.outcome for item in hit.constraint_evaluations
                    },
                    "unknown_fields": hit.structured_evidence.unknown_fields,
                    "supporting_snippets": hit.supporting_snippets,
                    "matched_exclude_terms": hit.matched_exclude_terms,
                    "profile_contribution": hit.profile_contribution,
                }
                for rank, hit in enumerate(result.hits, start=1)
            ],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
