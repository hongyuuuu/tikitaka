#!/usr/bin/env python3
"""Build or resume a dense artifact through Person 1's Embedder-compatible factory."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact
from tikitaka.retrieval.manifests import dense_manifest_as_dict


def _load_embedder(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedder-factory", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args()

    started = perf_counter()
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    loaded = perf_counter()
    embedder = _load_embedder(arguments.embedder_factory)
    manifest = build_dense_artifact(
        catalog,
        embedder,
        arguments.output,
        embedding_provider=arguments.provider,
        embedding_model=arguments.model,
        batch_size=arguments.batch_size,
    )
    finished = perf_counter()
    print(
        json.dumps(
            {
                "manifest": dense_manifest_as_dict(manifest),
                "timing_ms": {
                    "catalog_load": round((loaded - started) * 1_000, 3),
                    "artifact_build_or_verify": round((finished - loaded) * 1_000, 3),
                },
                "output_directory": str(arguments.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
