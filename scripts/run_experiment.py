#!/usr/bin/env python3
"""Run one explicit P5 arm and write reproducible tuning/held-out evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from tikitaka.config import CONTRACT_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.decision import phase4_arm, phase4_experiment_arms
from tikitaka.evaluation import (
    ExperimentConfig,
    P5ExperimentArm,
    SplitSpec,
    build_report,
    canonical_report_json,
    create_split,
    evaluate_samples,
    partition_samples,
)
from tikitaka.models.factory import gateway_from_env
from tikitaka.models.selector import AblationConfig, ModelSelector, SELECTIVE, pin_all
from tikitaka.orchestration.runtime import (
    DeterministicRuntimeConfig,
    RuntimeConfig,
    build_agent,
    build_deterministic_agent,
)
from tikitaka.retrieval import HybridConfig, HybridRetriever, load_catalog, load_dense_index
from tikitaka.state.query_builder import QueryBuilderConfig


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


def _load_factory(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def _not_run(split_name: str) -> dict[str, object]:
    return {"status": "not_run", "split": split_name}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-version", default="public-v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tuning-fraction", type=float, default=0.7)
    parser.add_argument(
        "--stage", choices=("tuning", "held_out", "both"), default="tuning",
        help="Held-out execution is opt-in so tuning cannot repeatedly probe it.",
    )
    parser.add_argument(
        "--confirm-held-out", action="store_true",
        help="Required for --stage held_out/both after tuning choices are frozen.",
    )
    parser.add_argument(
        "--retrieval-policy", choices=("sparse", "dense", "hybrid"), default="sparse"
    )
    parser.add_argument(
        "--generative-policy",
        choices=("deterministic", "api_always", "api_selective", "api_pinned"),
        default="deterministic",
    )
    parser.add_argument(
        "--decision-arm",
        choices=tuple(arm.name for arm in phase4_experiment_arms()),
        default="adaptive-deterministic",
    )
    parser.add_argument("--profile-weight", type=float)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedder-factory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.stage != "tuning" and not args.confirm_held_out:
        parser.error("--stage held_out/both requires --confirm-held-out")
    if args.retrieval_policy != "sparse" and (
        args.artifact is None or args.embedder_factory is None
    ):
        parser.error("dense/hybrid retrieval requires --artifact and --embedder-factory")

    arm = P5ExperimentArm(
        name=args.name,
        retrieval_policy=args.retrieval_policy,
        generative_policy=args.generative_policy,
        decision_arm=args.decision_arm,
        profile_weight=args.profile_weight,
    )
    decision = phase4_arm(arm.decision_arm)
    if arm.generative_policy == "deterministic" and decision.enable_llm_reranker:
        parser.error("an LLM reranker arm requires an API generative policy")

    samples = load_jsonl(args.dataset)
    split_spec = SplitSpec(args.split_version, args.seed, args.tuning_fraction)
    split_manifest = create_split(samples, split_spec)
    tuning, held_out = partition_samples(samples, split_manifest)
    checksum = _sha256(args.catalog)
    catalog = load_catalog(args.catalog)

    embedding_route_id = "none"
    index_id = f"catalog:{checksum}"
    fusion_parameters: tuple[tuple[str, float], ...] = ()
    query_config = QueryBuilderConfig(
        profile_weight=arm.selected_profile_weight,
        route_policy=arm.retrieval_policy,
    )
    if arm.retrieval_policy != "sparse":
        probe_embedder = _load_factory(args.embedder_factory)
        dense_index = load_dense_index(
            args.artifact,
            catalog,
            embedding_route_id=str(getattr(probe_embedder, "route_id", "")),
        )
        embedding_route_id = dense_index.manifest.route_id
        index_id = dense_index.manifest.index_id
        hybrid = HybridConfig()
        fusion_parameters = (
            ("rrf_k", float(hybrid.rrf_k)),
            ("sparse_weight", hybrid.sparse_weight),
            ("dense_weight", hybrid.dense_weight),
            ("buying_sparse_multiplier", hybrid.buying_sparse_multiplier),
            ("buying_dense_multiplier", hybrid.buying_dense_multiplier),
            ("browsing_sparse_multiplier", hybrid.browsing_sparse_multiplier),
            ("browsing_dense_multiplier", hybrid.browsing_dense_multiplier),
        )
        query_config = replace(
            query_config,
            embedding_route_id=embedding_route_id,
            index_id=index_id,
        )

    selection = None
    selector = None
    model_identity: dict[str, str] = {
        "routing_mode": "pinned",
        "generative_provider": "none",
        "generative_model": "heuristic/local",
        "reasoning_level": "none",
    }
    if arm.generative_policy != "deterministic":
        selection = gateway_from_env(allow_degraded=False)
        ablation = AblationConfig(profile_weight=arm.selected_profile_weight)
        if arm.generative_policy == "api_pinned":
            selector = ModelSelector(
                selection.route, pins=pin_all(selection.route), ablation=ablation
            )
        else:
            selector = ModelSelector(
                selection.route,
                ablation=ablation,
                thresholds=SELECTIVE if arm.generative_policy == "api_selective" else None,
            )
        model_identity = {
            "routing_mode": selector.routing_mode,
            "generative_provider": selection.route.provider,
            "generative_model": selection.route.model,
            "reasoning_level": selection.route.reasoning_level or "none",
        }

    config = ExperimentConfig(
        name=arm.name,
        config_version="p5-v1",
        prompt_version="intent-interpreter/1",
        schema_version=f"contracts-{CONTRACT_VERSION}/structured-{STRUCTURED_OUTPUT_SCHEMA_VERSION}",
        retrieval_policy=arm.retrieval_policy,
        embedding_route_id=embedding_route_id,
        index_id=index_id,
        reranker_route_id=decision.reranker_route_id,
        fusion_parameters=fusion_parameters,
        profile_weight=arm.selected_profile_weight,
        question_policy=decision.question_policy_id,
        seed=args.seed,
        split_version=args.split_version,
        catalog_checksum=checksum,
        code_revision=_revision(),
        ablation_parameters=arm.report_parameters(),
        **model_identity,
    )

    def agent_factory() -> object:
        retriever = None
        if arm.retrieval_policy != "sparse":
            embedder = _load_factory(args.embedder_factory)
            dense = load_dense_index(
                args.artifact, catalog, embedding_route_id=embedding_route_id
            )
            retriever = HybridRetriever(catalog, dense_index=dense, query_embedder=embedder)
        if arm.generative_policy == "deterministic":
            runtime = DeterministicRuntimeConfig(
                profile_weight=arm.selected_profile_weight,
                query_builder=query_config,
                decision=decision.response,
                ranking=decision.ranking,
            )
            return build_deterministic_agent(args.catalog, runtime, retriever=retriever)
        runtime_values = dict(decision.runtime_overrides)
        runtime_values.update(
            profile_weight=arm.selected_profile_weight,
            query_builder=query_config,
            allow_degraded=False,
            selector=selector,
        )
        runtime = RuntimeConfig(**runtime_values)
        built, _ = build_agent(
            args.catalog, runtime, model_selection=selection, retriever=retriever
        )
        return built

    catalog_ids, categories, products = catalog_index(args.catalog)
    tuning_result = (
        evaluate_samples(
            agent_factory, tuning, catalog_ids, categories, products, config, "tuning"
        )
        if args.stage in {"tuning", "both"}
        else _not_run("tuning")
    )
    held_out_result = (
        evaluate_samples(
            agent_factory, held_out, catalog_ids, categories, products, config, "held_out"
        )
        if args.stage in {"held_out", "both"}
        else _not_run("held_out")
    )
    report = build_report(config, split_manifest, tuning_result, held_out_result)
    output = args.output or Path("reports") / f"{arm.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_report_json(report), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "experiment_fingerprint": config.fingerprint,
        "arm_fingerprint": arm.fingerprint,
        "stage": args.stage,
        "tuning": tuning_result.get("metrics", tuning_result),
        "held_out": held_out_result.get("metrics", held_out_result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
