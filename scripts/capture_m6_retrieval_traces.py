#!/usr/bin/env python3
"""Capture label-free M6 retrieval traces over the frozen 50k catalog."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import load_dense_index
from tikitaka.retrieval.hybrid import HybridRetriever, HybridRetrievalResult
from tikitaka.retrieval.request import RetrievalConstraint, RetrievalRequest
from tikitaka.retrieval.text import DENSE_QUERY_SCHEMA_VERSION, PRODUCT_TEXT_SCHEMA_VERSION


SCHEMA_VERSION = "m6-retrieval-traces-v1"


def _load_embedder(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator:
        raise ValueError("embedder factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def trace_requests() -> dict[str, RetrievalRequest]:
    """Return visible, hand-authored scenarios with no evaluator labels."""

    return {
        "vague_trip_shoes": RetrievalRequest(
            text_query="shoes for a trip",
            mode="unknown",
            route_policy="sparse",
        ),
        "explicit_constraint_narrowing": RetrievalRequest(
            text_query="comfortable water-resistant shoes for long walks",
            constraints=(
                RetrievalConstraint("category", ("shoes",)),
                RetrievalConstraint("feature", ("water-resistant", "comfortable")),
                RetrievalConstraint("use_case", ("long walking",)),
                RetrievalConstraint(
                    "budget", (Decimal("80"),), strength="hard", operator="lte"
                ),
            ),
            mode="buying",
            route_policy="sparse",
        ),
        "intent_override_before": RetrievalRequest(
            text_query="red leather boots",
            constraints=(
                RetrievalConstraint("category", ("boots",)),
                RetrievalConstraint("material", ("leather",)),
                RetrievalConstraint("color", ("red",)),
            ),
            mode="buying",
            intent_version=1,
            route_policy="sparse",
        ),
        "intent_override_after": RetrievalRequest(
            text_query="running shoes",
            constraints=(
                RetrievalConstraint("category", ("shoes",)),
                RetrievalConstraint("feature", ("running",)),
            ),
            no_preference=frozenset({"color"}),
            mode="buying",
            intent_version=2,
            route_policy="sparse",
        ),
        "boundary_no_material_preference": RetrievalRequest(
            text_query="comfortable shoes",
            constraints=(
                RetrievalConstraint("category", ("shoes",)),
                RetrievalConstraint("feature", ("comfortable",)),
            ),
            no_preference=frozenset({"material"}),
            mode="browsing",
            route_policy="sparse",
        ),
    }


def _json_value(value: object) -> object:
    return str(value) if isinstance(value, Decimal) else value


def _request_payload(request: RetrievalRequest) -> dict[str, object]:
    return {
        "text_query": request.text_query,
        "mode": request.mode,
        "intent_version": request.intent_version,
        "no_preference": sorted(request.no_preference),
        "profile_terms": list(request.profile_terms),
        "profile_weight": request.profile_weight,
        "route_policy": request.route_policy,
        "embedding_route_id": request.embedding_route_id,
        "index_id": request.index_id,
        "constraints": [
            {
                "attribute": item.attribute,
                "values": [_json_value(value) for value in item.values],
                "polarity": item.polarity,
                "strength": item.strength,
                "operator": item.operator,
                "needs_revalidation": item.needs_revalidation,
            }
            for item in request.constraints
        ],
    }


def _result_payload(result: HybridRetrievalResult) -> dict[str, object]:
    diagnostics = result.diagnostics
    return {
        "diagnostics": {
            "requested_route": diagnostics.requested_route,
            "executed_route": diagnostics.executed_route,
            "route_failures": list(diagnostics.route_failures),
            "sparse_candidates": diagnostics.sparse_candidates,
            "dense_candidates": diagnostics.dense_candidates,
            "fused_candidates": diagnostics.fused_candidates,
            "hard_filtered_candidates": diagnostics.hard_filtered_candidates,
            "returned_candidates": diagnostics.returned_candidates,
            "intent_version": diagnostics.intent_version,
            "route_overlap": dict(diagnostics.route_overlap),
            "top_score_margin": diagnostics.top_score_margin,
            "top_score_concentration": diagnostics.top_score_concentration,
            "effective_candidate_count": diagnostics.effective_candidate_count,
            "missing_attribute_rates": dict(diagnostics.missing_attribute_rates),
            "constraint_outcome_counts": {
                name: dict(counts)
                for name, counts in diagnostics.constraint_outcome_counts.items()
            },
            "route_timings_ms": dict(diagnostics.route_timings_ms),
            "manifest_ids": dict(diagnostics.manifest_ids),
        },
        "top_10": [
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
                "supporting_snippets": list(hit.supporting_snippets),
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
    }


def _assert_override_is_reduced(requests: dict[str, RetrievalRequest]) -> None:
    before = json.dumps(_request_payload(requests["intent_override_before"])).casefold()
    after = json.dumps(_request_payload(requests["intent_override_after"])).casefold()
    if "leather" not in before or "red" not in before:
        raise RuntimeError("override fixture does not contain the stale constraints")
    if "leather" in after or '"values": ["red"]' in after:
        raise RuntimeError("new-intent request retained a stale dependent constraint")
    if requests["intent_override_after"].intent_version != 2:
        raise RuntimeError("new-intent request did not increment intent_version")


def capture(
    *,
    catalog_path: Path,
    hybrid_artifact: Path | None = None,
    embedder_factory: str | None = None,
) -> dict[str, object]:
    catalog = load_catalog(catalog_path, expected_count=50_000)
    requests = trace_requests()
    _assert_override_is_reduced(requests)
    cases: dict[str, object] = {}
    with HybridRetriever(catalog) as sparse:
        for name, request in requests.items():
            result = sparse.retrieve(request, limit=10)
            cases[name] = {
                "purpose": {
                    "vague_trip_shoes": "vague request becomes an observable search plan",
                    "explicit_constraint_narrowing": "constraints narrow and explain the pool",
                    "intent_override_before": "old intent before dependency-aware clearing",
                    "intent_override_after": "new intent excludes stale leather/red/boots state",
                    "boundary_no_material_preference": "material is deliberately omitted after no-preference",
                }[name],
                "request": _request_payload(request),
                "result": _result_payload(result),
            }

    comparison: dict[str, object] = {
        "status": "pending_fixture_inputs",
        "production_quality_claim": False,
    }
    if hybrid_artifact is not None or embedder_factory is not None:
        if hybrid_artifact is None or embedder_factory is None:
            raise ValueError("hybrid artifact and embedder factory must be supplied together")
        embedder = _load_embedder(embedder_factory)
        index = load_dense_index(
            hybrid_artifact,
            catalog,
            embedding_route_id=str(getattr(embedder, "route_id", "")),
        )
        base = requests["explicit_constraint_narrowing"]
        hybrid_request = RetrievalRequest(
            text_query=base.text_query,
            constraints=base.constraints,
            mode=base.mode,
            intent_version=base.intent_version,
            no_preference=base.no_preference,
            route_policy="hybrid",
            embedding_route_id=index.manifest.route_id,
            index_id=index.manifest.index_id,
        )
        with HybridRetriever(
            catalog,
            dense_index=index,
            query_embedder=embedder,
        ) as hybrid:
            hybrid_result = hybrid.retrieve(hybrid_request, limit=10)
        comparison = {
            "status": "fixture_mechanics_only",
            "production_quality_claim": False,
            "warning": (
                "Deterministic keyword fixture embeddings prove routing, identity, "
                "fusion and evidence plumbing only; they do not estimate production quality."
            ),
            "query": base.text_query,
            "sparse": cases["explicit_constraint_narrowing"]["result"],
            "hybrid": _result_payload(hybrid_result),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "catalog": {
            "rows": len(catalog),
            "sha256": catalog.identity.source_sha256,
            "ordered_parent_asin_sha256": catalog.identity.ordered_parent_asin_sha256,
        },
        "label_policy": {
            "hand_authored_visible_queries": True,
            "ground_truth_used": False,
            "scenario_labels_used": False,
            "user_profile_used": False,
        },
        "frozen_versions": {
            "product_text_schema": PRODUCT_TEXT_SCHEMA_VERSION,
            "dense_query_schema": DENSE_QUERY_SCHEMA_VERSION,
        },
        "cases": cases,
        "sparse_vs_hybrid": comparison,
        "production_dense_evidence": {
            "status": "pending_production_1024_index",
            "dimensions": 1024,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hybrid-artifact", type=Path)
    parser.add_argument("--embedder-factory")
    arguments = parser.parse_args()
    payload = capture(
        catalog_path=arguments.catalog,
        hybrid_artifact=arguments.hybrid_artifact,
        embedder_factory=arguments.embedder_factory,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
