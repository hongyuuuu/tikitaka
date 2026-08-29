#!/usr/bin/env python3
"""Benchmark pinned retrieval routes on Person 4-provided split-aware cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tikitaka.contracts import Usage
from tikitaka.models.usage import merge as merge_usage
from tikitaka.retrieval.adapters import contract_candidate
from tikitaka.retrieval.benchmark import (
    evaluate_retrieval_route,
    load_retrieval_benchmark_cases,
)
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import load_dense_index
from tikitaka.retrieval.embedding import embedding_usage_as_dict
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.manifests import dense_manifest_as_dict
from tikitaka.retrieval.retriever import SparseStructuredRetriever


def _factory(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _routes(value: str) -> tuple[str, ...]:
    routes = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    allowed = {"sparse", "dense", "hybrid"}
    if not routes or set(routes).difference(allowed):
        raise argparse.ArgumentTypeError("routes must contain sparse, dense, and/or hybrid")
    return routes


def _ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted(set(int(item.strip()) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from error
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("K values must be positive")
    return ks


def _take_usage(embedder: object | None) -> Usage | None:
    take = None if embedder is None else getattr(embedder, "take_usage", None)
    if not callable(take):
        return None
    usage = take()
    if not isinstance(usage, Usage):
        raise TypeError("embedder.take_usage() must return canonical Usage")
    if usage.calls == 0 and not usage.cache_hit:
        return None
    return usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--routes", type=_routes, default=("sparse", "dense", "hybrid"))
    parser.add_argument("--ks", type=_ks, default=(10, 50, 100, 200))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedder-factory")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-route-degradation",
        action="store_true",
        help="Report dense failures instead of failing a model-comparison run.",
    )
    arguments = parser.parse_args()

    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    cases = load_retrieval_benchmark_cases(arguments.cases, valid_ids=catalog.ids)
    needs_dense = any(route in {"dense", "hybrid"} for route in arguments.routes)
    if needs_dense and (arguments.artifact is None or arguments.embedder_factory is None):
        parser.error("dense/hybrid routes require --artifact and --embedder-factory")

    embedder = None
    dense_index = None
    if needs_dense:
        embedder = _factory(arguments.embedder_factory)
        dense_index = load_dense_index(
            arguments.artifact,
            catalog,
            embedding_route_id=str(getattr(embedder, "route_id", "")),
        )

    results: dict[str, object] = {}
    executions: dict[str, dict[str, Counter[str]]] = {}
    total_usage: Usage | None = None
    sparse = SparseStructuredRetriever(catalog)
    hybrid = (
        HybridRetriever(catalog, dense_index=dense_index, query_embedder=embedder)
        if needs_dense
        else None
    )
    try:
        for route in arguments.routes:
            _take_usage(embedder)
            if route == "sparse":
                search = lambda request, limit: sparse.search(request, limit)
            else:
                assert hybrid is not None and dense_index is not None
                executed_counts: Counter[str] = Counter()
                failure_counts: Counter[str] = Counter()
                executions[route] = {
                    "executed_routes": executed_counts,
                    "failure_codes": failure_counts,
                }

                def search(request, limit, selected=route):
                    pinned = replace(
                        request,
                        route_policy=selected,
                        embedding_route_id=dense_index.manifest.route_id,
                        index_id=dense_index.manifest.index_id,
                    )
                    result = hybrid.retrieve(pinned, limit=limit)
                    executed_counts[result.diagnostics.executed_route] += 1
                    failure_counts.update(result.diagnostics.route_failures)
                    return [contract_candidate(hit) for hit in result.hits]

            results[route] = evaluate_retrieval_route(
                cases,
                search,
                valid_ids=catalog.ids,
                ks=arguments.ks,
            )
            route_usage = _take_usage(embedder)
            results[route]["embedding_usage"] = (
                None if route_usage is None else embedding_usage_as_dict(route_usage)
            )
            if route_usage is not None:
                total_usage = (
                    route_usage if total_usage is None else merge_usage(total_usage, route_usage)
                )
            if route != "sparse":
                route_execution = executions[route]
                results[route]["execution"] = {
                    name: dict(sorted(counts.items()))
                    for name, counts in route_execution.items()
                }
                expected_execution = route
                degraded_executions = sum(
                    count
                    for executed_route, count in route_execution["executed_routes"].items()
                    if executed_route != expected_execution
                )
                results[route]["execution"]["degraded_case_count"] = degraded_executions
                if (
                    route_execution["failure_codes"] or degraded_executions
                ) and not arguments.allow_route_degradation:
                    details = set(route_execution["failure_codes"])
                    details.update(
                        "executed_as_" + executed_route
                        for executed_route in route_execution["executed_routes"]
                        if executed_route != expected_execution
                    )
                    raise RuntimeError(
                        f"{route} benchmark degraded: " + ", ".join(sorted(details))
                    )
    finally:
        sparse.close()
        if hybrid is not None:
            hybrid.close()

    usage = total_usage
    if usage is None and embedder is not None:
        cumulative = getattr(embedder, "usage", None)
        usage = cumulative if isinstance(cumulative, Usage) else None
    payload = {
        "benchmark_schema_version": "retrieval-benchmark-v1",
        "case_file": str(arguments.cases),
        "case_file_sha256": _sha256(arguments.cases),
        "catalog_checksum": catalog.identity.source_sha256,
        "catalog_row_count": len(catalog),
        "routes": arguments.routes,
        "ks": arguments.ks,
        "retrieval_configuration": {
            "sparse_index": sparse.sparse.manifest.as_dict(),
            "sparse_retrieval": asdict(sparse.config),
            "hybrid": None if hybrid is None else asdict(hybrid.config),
        },
        "dense_manifest": (
            None if dense_index is None else dense_manifest_as_dict(dense_index.manifest)
        ),
        "embedding_usage": (
            embedding_usage_as_dict(usage) if isinstance(usage, Usage) else None
        ),
        "results": results,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
