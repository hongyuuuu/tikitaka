#!/usr/bin/env python3
"""Run a named deterministic public-set experiment and write its full report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from starter.agent import Agent
from tikitaka.config import CONTRACT_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.evaluation import (
    ExperimentConfig,
    SplitSpec,
    build_report,
    canonical_report_json,
    create_split,
    evaluate_samples,
    partition_samples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--split-version", default="public-v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tuning-fraction", type=float, default=0.7)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output or f"reports/{args.name}.json")
    samples = load_jsonl(dataset_path)
    spec = SplitSpec(args.split_version, args.seed, args.tuning_fraction)
    manifest = create_split(samples, spec)
    tuning, held_out = partition_samples(samples, manifest)
    checksum = _sha256(catalog_path)
    config = ExperimentConfig(
        name=args.name,
        config_version="1.0.0",
        prompt_version="deterministic-scaffold-v1",
        schema_version=f"contracts-{CONTRACT_VERSION}/structured-{STRUCTURED_OUTPUT_SCHEMA_VERSION}",
        routing_mode="pinned",
        generative_provider="none",
        generative_model="deterministic-scaffold",
        reasoning_level="none",
        retrieval_policy="sparse",
        embedding_route_id="none",
        index_id=f"catalog:{checksum}",
        reranker_route_id="deterministic",
        fusion_parameters=(),
        profile_weight=0.0,
        question_policy="scaffold-v1",
        seed=args.seed,
        split_version=args.split_version,
        catalog_checksum=checksum,
        code_revision=_revision(),
    )
    catalog_ids, categories, products = catalog_index(catalog_path)
    factory = lambda: Agent(catalog_path)
    tuning_result = evaluate_samples(factory, tuning, catalog_ids, categories, products, config, "tuning")
    held_out_result = evaluate_samples(factory, held_out, catalog_ids, categories, products, config, "held_out")
    report = build_report(config, manifest, tuning_result, held_out_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_report_json(report), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "experiment_fingerprint": config.fingerprint,
        "tuning": tuning_result["metrics"],
        "held_out": held_out_result["metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
