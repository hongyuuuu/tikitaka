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

from tikitaka.contracts import Usage
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact
from tikitaka.retrieval.embedding import embedding_usage_as_dict
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


def _identity(embedder: object, name: str, requested: str | None) -> str:
    declared = getattr(embedder, name, None)
    if declared is not None and (not isinstance(declared, str) or not declared.strip()):
        raise ValueError(f"embedder {name} must be a non-empty string when declared")
    declared = None if declared is None else declared.strip()
    requested = None if requested is None else requested.strip()
    if requested == "":
        raise ValueError(f"--{name} must be non-empty")
    if declared is not None and requested is not None and declared != requested:
        raise ValueError(f"--{name} does not match the embedder's declared {name}")
    selected = declared or requested
    if selected is None:
        raise ValueError(
            f"--{name} is required when the embedder does not declare a {name} property"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedder-factory", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args()

    started = perf_counter()
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    loaded = perf_counter()
    embedder = _load_embedder(arguments.embedder_factory)
    provider = _identity(embedder, "provider", arguments.provider)
    model = _identity(embedder, "model", arguments.model)
    manifest = build_dense_artifact(
        catalog,
        embedder,
        arguments.output,
        embedding_provider=provider,
        embedding_model=model,
        batch_size=arguments.batch_size,
    )
    finished = perf_counter()
    usage = getattr(embedder, "usage", None)
    print(
        json.dumps(
            {
                "manifest": dense_manifest_as_dict(manifest),
                "timing_ms": {
                    "catalog_load": round((loaded - started) * 1_000, 3),
                    "artifact_build_or_verify": round((finished - loaded) * 1_000, 3),
                },
                "output_directory": str(arguments.output.resolve()),
                "embedding_usage": (
                    embedding_usage_as_dict(usage) if isinstance(usage, Usage) else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
