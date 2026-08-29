#!/usr/bin/env python3
"""Run M4 sparse retrieval variants through the full deterministic public simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from tikitaka.config import CONTRACT_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.contracts import RoutePolicy
from tikitaka.decision import ResponsePolicy
from tikitaka.evaluation import (
    ExperimentConfig,
    SplitSpec,
    build_report,
    canonical_report_json,
    create_split,
    evaluate_samples,
    partition_samples,
)
from tikitaka.evaluation.ablations import compare_reports
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.orchestration.runtime import VisibleMessageInterpreter
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent
from tikitaka.ranking import DeterministicRanker
from tikitaka.retrieval.catalog import ProductCatalog, load_catalog
from tikitaka.retrieval.retriever import SparseStructuredRetriever
from tikitaka.retrieval.tuning import RetrievalSweepSpec, RetrievalVariant, load_retrieval_sweep_spec
from tikitaka.state.query_builder import ActiveQueryBuilder, QueryBuilderConfig
from tikitaka.state.reducer import StateReducer
from tikitaka.state.session import SessionState, new_session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _flatten_numeric(prefix: str, values: Mapping[str, object]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for name, value in sorted(values.items()):
        if isinstance(value, bool):
            result.append((f"{prefix}.{name}", float(value)))
        elif isinstance(value, (int, float)):
            result.append((f"{prefix}.{name}", float(value)))
    return result


def _experiment_config(
    variant: RetrievalVariant,
    spec: RetrievalSweepSpec,
    *,
    split_version: str,
    seed: int,
    catalog_checksum: str,
    code_revision: str,
) -> ExperimentConfig:
    parameters = [
        *_flatten_numeric("sparse", variant.to_dict()["sparse"]),
        *_flatten_numeric("ranking", variant.to_dict()["ranking"]),
        ("candidate_limit", float(variant.candidate_limit)),
    ]
    return ExperimentConfig(
        name=f"m4-{variant.variant_id}",
        config_version=spec.schema_version,
        prompt_version="visible-message-deterministic-v1",
        schema_version=(
            f"contracts-{CONTRACT_VERSION}/structured-{STRUCTURED_OUTPUT_SCHEMA_VERSION}"
        ),
        routing_mode="pinned",
        generative_provider="none",
        generative_model="deterministic-visible-message",
        reasoning_level="none",
        retrieval_policy="sparse",
        embedding_route_id="none",
        index_id=f"catalog:{catalog_checksum}",
        reranker_route_id="deterministic",
        fusion_parameters=tuple(parameters),
        profile_weight=variant.request_profile_weight or 0.0,
        question_policy="response-policy-v1",
        seed=seed,
        split_version=split_version,
        catalog_checksum=catalog_checksum,
        code_revision=code_revision,
    )


def _build_agent(
    catalog: ProductCatalog,
    variant: RetrievalVariant,
) -> ShoppingAgent[SessionState]:
    profile_weight = variant.request_profile_weight or 0.0
    retriever = SparseStructuredRetriever(
        catalog,
        sparse_config=variant.sparse,
        retrieval_config=variant.ranking,
    )
    sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
    return ShoppingAgent(
        sessions=sessions,
        reducer=StateReducer(),
        interpreter=VisibleMessageInterpreter(HeuristicInterpreter()),
        query_builder=ActiveQueryBuilder(
            QueryBuilderConfig(
                profile_weight=profile_weight,
                route_policy=RoutePolicy.SPARSE,
            )
        ),
        retriever=retriever,
        decision_policy=ResponsePolicy(),
        reranker=DeterministicRanker(),
        catalog_ids=catalog.ids,
        candidate_limit=variant.candidate_limit,
    )


def _metrics(report: Mapping[str, object], split: str) -> Mapping[str, object]:
    return report["results"][split]["metrics"]  # type: ignore[index,return-value]


def _scenarios(report: Mapping[str, object], split: str) -> Mapping[str, Mapping[str, object]]:
    return report["results"][split]["scenario_metrics"]  # type: ignore[index,return-value]


def _selection_key(report: Mapping[str, object]) -> tuple[object, ...]:
    metrics = _metrics(report, "tuning")
    return (
        -float(metrics["hit_rate_at_10"]),
        -float(metrics["mrr"]),
        float(metrics["mttc"]),
        -float(metrics["technical_score"]),
        str(report["experiment"]["configuration"]["name"]),  # type: ignore[index]
    )


def _variant_id(report: Mapping[str, object]) -> str:
    name = str(report["experiment"]["configuration"]["name"])  # type: ignore[index]
    return name.removeprefix("m4-")


def _select(
    spec: RetrievalSweepSpec,
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_id = {_variant_id(report): report for report in reports}
    baseline = by_id[spec.baseline_variant_id]
    baseline_scenarios = _scenarios(baseline, "tuning")
    eligible: list[Mapping[str, object]] = []
    guards: dict[str, object] = {}
    for variant_id, report in sorted(by_id.items()):
        scenarios = _scenarios(report, "tuning")
        deltas = {
            scenario: round(
                float(scenarios[scenario]["hit_rate_at_10"])
                - float(baseline_scenarios[scenario]["hit_rate_at_10"]),
                12,
            )
            for scenario in sorted(baseline_scenarios)
        }
        collapsed = [
            scenario
            for scenario, delta in deltas.items()
            if delta < -spec.max_scenario_hit_rate_drop
        ]
        guards[variant_id] = {
            "eligible": not collapsed,
            "scenario_hit_rate_deltas": deltas,
            "collapsed_scenarios": collapsed,
        }
        if not collapsed:
            eligible.append(report)
    if not eligible:
        raise RuntimeError("no sparse runtime variant passes the scenario-collapse guard")
    ranked = sorted(eligible, key=_selection_key)
    winner = ranked[0]
    winner_id = _variant_id(winner)
    return {
        "selection_basis": "tuning_only",
        "selected_variant_id": winner_id,
        "eligible_tuning_ranking": [_variant_id(report) for report in ranked],
        "scenario_collapse_guards": guards,
        "heldout_used_for_selection": False,
        "heldout_confirmation": {
            "variant_id": winner_id,
            "overall_delta_from_baseline": compare_reports(
                baseline,
                winner,
                ["fusion_parameters", "profile_weight"],
                split_name="held_out",
            )["metric_deltas"],
            "scenario_hit_rate_deltas": {
                scenario: round(
                    float(_scenarios(winner, "held_out")[scenario]["hit_rate_at_10"])
                    - float(
                        _scenarios(baseline, "held_out")[scenario]["hit_rate_at_10"]
                    ),
                    12,
                )
                for scenario in sorted(_scenarios(baseline, "held_out"))
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--split-version", default="public-v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tuning-fraction", type=float, default=0.7)
    arguments = parser.parse_args()

    revision, dirty = _revision()
    if dirty:
        parser.error("public M4 evidence requires a clean committed worktree")
    spec = load_retrieval_sweep_spec(arguments.spec)
    variants = tuple(variant for variant in spec.variants if variant.route == "sparse")
    if spec.baseline_variant_id not in {variant.variant_id for variant in variants}:
        parser.error("the M4 baseline must be a sparse variant")
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    raw_samples = load_jsonl(arguments.dataset)
    split_spec = SplitSpec(
        arguments.split_version,
        arguments.seed,
        arguments.tuning_fraction,
    )
    split = create_split(raw_samples, split_spec)
    tuning, held_out = partition_samples(raw_samples, split)
    catalog_ids, categories, products = catalog_index(arguments.catalog)
    catalog_checksum = _sha256(arguments.catalog)
    reports: list[dict[str, object]] = []
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        config = _experiment_config(
            variant,
            spec,
            split_version=arguments.split_version,
            seed=arguments.seed,
            catalog_checksum=catalog_checksum,
            code_revision=revision,
        )
        created_agents: list[ShoppingAgent[SessionState]] = []

        def factory(selected=variant):
            agent = _build_agent(catalog, selected)
            created_agents.append(agent)
            return agent

        try:
            tuning_result = evaluate_samples(
                factory,
                tuning,
                catalog_ids,
                categories,
                products,
                config,
                "tuning",
            )
            heldout_result = evaluate_samples(
                factory,
                held_out,
                catalog_ids,
                categories,
                products,
                config,
                "held_out",
            )
        finally:
            for agent in created_agents:
                agent.close()
        report = build_report(config, split, tuning_result, heldout_result)
        reports.append(report)
        path = arguments.output_directory / f"{variant.variant_id}.json"
        path.write_text(canonical_report_json(report), encoding="utf-8")

    baseline = next(
        report for report in reports if _variant_id(report) == spec.baseline_variant_id
    )
    comparisons = {
        _variant_id(report): {
            "tuning": compare_reports(
                baseline,
                report,
                ["fusion_parameters", "profile_weight"],
                split_name="tuning",
            ),
            "held_out": compare_reports(
                baseline,
                report,
                ["fusion_parameters", "profile_weight"],
                split_name="held_out",
            ),
        }
        for report in reports
    }
    summary = {
        "runtime_sweep_schema_version": "m4-sparse-runtime-v1",
        "code_revision": revision,
        "code_dirty": False,
        "catalog_checksum": catalog_checksum,
        "dataset_checksum": _sha256(arguments.dataset),
        "sweep_spec_fingerprint": spec.fingerprint,
        "split": split.to_dict(),
        "executed_variant_ids": [_variant_id(report) for report in reports],
        "deferred_variant_ids": [
            variant.variant_id for variant in spec.variants if variant.route != "sparse"
        ],
        "deferred_reason": (
            "Dense, hybrid, and automatic runtime variants require Person 1's "
            "production embedding route and Person 4's API composition root."
        ),
        "selection": _select(spec, reports),
        "comparisons_from_baseline": comparisons,
        "report_files": {
            _variant_id(report): f"{_variant_id(report)}.json" for report in reports
        },
    }
    summary_path = arguments.output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(summary_path),
                "executed_variant_count": len(reports),
                "selection": summary["selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
