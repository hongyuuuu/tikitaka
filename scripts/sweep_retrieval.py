#!/usr/bin/env python3
"""Run a strict M4 retrieval sweep and write one reproducible comparison report."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tikitaka.retrieval.benchmark import load_retrieval_benchmark_cases
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import load_dense_index
from tikitaka.retrieval.manifests import dense_manifest_as_dict
from tikitaka.retrieval.tuning import (
    EVIDENCE_TIERS,
    build_retrieval_sweep_report,
    load_retrieval_sweep_spec,
    run_retrieval_variant,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> tuple[str, bool]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unknown", bool(status.stdout.strip())


def _factory(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def _ks(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted(set(int(item.strip()) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "K values must be comma-separated integers"
        ) from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("K values must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedder-factory")
    parser.add_argument("--ks", type=_ks, default=(10, 50, 100, 200))
    parser.add_argument(
        "--evidence-tier",
        choices=sorted(EVIDENCE_TIERS),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-route-degradation",
        action="store_true",
        help="Record route fallback instead of failing the comparison.",
    )
    arguments = parser.parse_args()

    output_path = arguments.output.resolve()
    if (
        arguments.evidence_tier == "public-development"
        and output_path.is_relative_to(PROJECT_ROOT)
    ):
        parser.error("public M4 output must be outside the source repository")

    spec = load_retrieval_sweep_spec(arguments.spec)
    if spec.selection_k not in arguments.ks:
        parser.error("--ks must contain the sweep selection_k")
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    cases = load_retrieval_benchmark_cases(arguments.cases, valid_ids=catalog.ids)
    needs_dense = any(variant.requires_dense for variant in spec.variants)
    if needs_dense and (arguments.artifact is None or arguments.embedder_factory is None):
        parser.error("dense/auto/hybrid variants require --artifact and --embedder-factory")

    embedder = _factory(arguments.embedder_factory) if needs_dense else None
    dense_index = (
        load_dense_index(
            arguments.artifact,
            catalog,
            embedding_route_id=str(getattr(embedder, "route_id", "")),
        )
        if needs_dense
        else None
    )
    reports = [
        run_retrieval_variant(
            variant,
            cases,
            catalog,
            ks=arguments.ks,
            dense_index=dense_index,
            query_embedder=embedder,
            allow_route_degradation=arguments.allow_route_degradation,
        )
        for variant in spec.variants
    ]
    revision, dirty = _revision()
    report = build_retrieval_sweep_report(
        spec,
        reports,
        evidence_tier=arguments.evidence_tier,
        code_revision=revision,
        code_dirty=dirty,
        case_file=str(arguments.cases),
        case_file_sha256=_sha256(arguments.cases),
        catalog=catalog,
        dense_manifest=(
            None
            if dense_index is None
            else dense_manifest_as_dict(dense_index.manifest)
        ),
        dense_backend=None if dense_index is None else dense_index.backend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sweep_spec_fingerprint": spec.fingerprint,
                "variant_count": len(reports),
                "selection": report["selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
