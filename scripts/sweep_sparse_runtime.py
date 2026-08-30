#!/usr/bin/env python3
"""Run M4 retrieval variants through the full deterministic public simulator.

With no dense artifact, the command remains a sparse-only dependency-safe
runner. Supplying an artifact and matching embedder executes the complete
sparse, dense, hybrid, and automatic grid and fails on any route degradation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from tikitaka.config import CONTRACT_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.contracts import RoutePolicy, Usage
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
from tikitaka.retrieval.adapters import contract_candidate
from tikitaka.retrieval.catalog import ProductCatalog, load_catalog
from tikitaka.retrieval.dense import DenseIndex, load_dense_index
from tikitaka.retrieval.embedding import embedding_usage_as_dict
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.retriever import SparseStructuredRetriever
from tikitaka.retrieval.tuning import RetrievalSweepSpec, RetrievalVariant, load_retrieval_sweep_spec
from tikitaka.state.query_builder import ActiveQueryBuilder, QueryBuilderConfig
from tikitaka.state.reducer import StateReducer
from tikitaka.state.session import SessionState, new_session


RUNTIME_COMPARISON_VARIABLES = (
    "embedding_route_id",
    "fusion_parameters",
    "index_id",
    "profile_weight",
    "retrieval_policy",
)


def _load_embedder(specification: str) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder factory must use module.path:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"embedder factory is not callable: {specification}")
    return factory()


def _take_embedding_usage(embedder: object | None) -> Usage | None:
    take = None if embedder is None else getattr(embedder, "take_usage", None)
    if not callable(take):
        return None
    usage = take()
    if not isinstance(usage, Usage):
        raise TypeError("embedder.take_usage() must return canonical Usage")
    if usage.calls == 0 and not usage.cache_hit:
        return None
    return usage


class _RouteTracker:
    def __init__(self) -> None:
        self.attempts = 0
        self.successful_calls = 0
        self.executed: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.exceptions: Counter[str] = Counter()

    def start(self) -> None:
        self.attempts += 1

    def record(self, diagnostics: object) -> None:
        self.successful_calls += 1
        self.executed[str(getattr(diagnostics, "executed_route"))] += 1
        self.failures.update(getattr(diagnostics, "route_failures"))

    def record_exception(self, error: Exception) -> None:
        self.exceptions[type(error).__name__] += 1

    def to_dict(self, expected_route: str) -> dict[str, object]:
        allowed_routes = (
            {"sparse"}
            if expected_route == "sparse"
            else {"dense", "sparse"}
            if expected_route == "dense"
            else {"hybrid", "dense", "sparse"}
        )
        degraded = sum(
            count
            for route, count in self.executed.items()
            if route not in allowed_routes
        )
        return {
            "attempts": self.attempts,
            "successful_calls": self.successful_calls,
            "expected_route": expected_route,
            "allowed_execution_routes": sorted(allowed_routes),
            "executed_routes": dict(sorted(self.executed.items())),
            "failure_codes": dict(sorted(self.failures.items())),
            "exception_types": dict(sorted(self.exceptions.items())),
            "degraded_call_count": degraded,
        }


class _ObservedHybridRetriever:
    def __init__(self, retriever: HybridRetriever, tracker: _RouteTracker) -> None:
        self._retriever = retriever
        self._tracker = tracker

    def search(self, plan: object, limit: int):
        self._tracker.start()
        try:
            result = self._retriever.retrieve(plan, limit=limit)
        except Exception as error:
            self._tracker.record_exception(error)
            raise
        self._tracker.record(result.diagnostics)
        return [contract_candidate(hit) for hit in result.hits]

    def close(self) -> None:
        self._retriever.close()


class _ObservedSparseRetriever:
    def __init__(
        self,
        retriever: SparseStructuredRetriever,
        tracker: _RouteTracker,
    ) -> None:
        self._retriever = retriever
        self._tracker = tracker
        self.sparse = retriever.sparse

    def search(self, plan: object, limit: int):
        self._tracker.start()
        try:
            candidates = self._retriever.search(plan, limit)
        except Exception as error:
            self._tracker.record_exception(error)
            raise
        self._tracker.successful_calls += 1
        self._tracker.executed["sparse"] += 1
        return candidates

    def close(self) -> None:
        self._retriever.close()


def _validated_route_execution(
    variant_id: str,
    expected_route: str,
    trackers: Sequence[_RouteTracker],
) -> dict[str, object]:
    combined = _RouteTracker()
    for tracker in trackers:
        combined.attempts += tracker.attempts
        combined.successful_calls += tracker.successful_calls
        combined.executed.update(tracker.executed)
        combined.failures.update(tracker.failures)
        combined.exceptions.update(tracker.exceptions)
    execution = combined.to_dict(expected_route)
    incomplete = execution["attempts"] != execution["successful_calls"]
    if (
        not execution["attempts"]
        or execution["failure_codes"]
        or execution["exception_types"]
        or execution["degraded_call_count"]
        or incomplete
    ):
        raise RuntimeError(
            f"variant {variant_id} degraded: "
            + json.dumps(execution, sort_keys=True)
        )
    return execution


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
    dense_index: DenseIndex | None = None,
) -> ExperimentConfig:
    parameters = [
        *_flatten_numeric("sparse", variant.to_dict()["sparse"]),
        *_flatten_numeric("ranking", variant.to_dict()["ranking"]),
        *_flatten_numeric("hybrid", variant.to_dict()["hybrid"]),
        ("candidate_limit", float(variant.candidate_limit)),
    ]
    embedding_route_id = (
        "none" if dense_index is None else dense_index.manifest.route_id
    )
    index_id = (
        f"catalog:{catalog_checksum}"
        if dense_index is None
        else dense_index.manifest.index_id
    )
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
        retrieval_policy=variant.route,
        embedding_route_id=embedding_route_id,
        index_id=index_id,
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
    *,
    dense_index: DenseIndex | None = None,
    query_embedder: object | None = None,
    route_tracker: _RouteTracker | None = None,
) -> ShoppingAgent[SessionState]:
    profile_weight = variant.request_profile_weight or 0.0
    if variant.requires_dense:
        if dense_index is None or query_embedder is None:
            raise ValueError(
                f"variant {variant.variant_id} requires a dense index and embedder"
            )
        tracker = route_tracker or _RouteTracker()
        retriever = _ObservedHybridRetriever(
            HybridRetriever(
                catalog,
                dense_index=dense_index,
                query_embedder=query_embedder,
                sparse_config=variant.sparse,
                config=variant.hybrid,
            ),
            tracker,
        )
        embedding_route_id = dense_index.manifest.route_id
        index_id = dense_index.manifest.index_id
        route_policy = RoutePolicy(variant.route)
    else:
        tracker = route_tracker or _RouteTracker()
        retriever = _ObservedSparseRetriever(
            SparseStructuredRetriever(
                catalog,
                sparse_config=variant.sparse,
                retrieval_config=variant.ranking,
            ),
            tracker,
        )
        embedding_route_id = None
        index_id = None
        route_policy = RoutePolicy.SPARSE
    sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
    return ShoppingAgent(
        sessions=sessions,
        reducer=StateReducer(),
        interpreter=VisibleMessageInterpreter(HeuristicInterpreter()),
        query_builder=ActiveQueryBuilder(
            QueryBuilderConfig(
                profile_weight=profile_weight,
                route_policy=route_policy,
                embedding_route_id=embedding_route_id,
                index_id=index_id,
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
    *,
    evidence_tier: str = "public-development",
) -> dict[str, object]:
    if evidence_tier not in {"fixture", "public-development"}:
        raise ValueError(f"unsupported evidence tier: {evidence_tier}")
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
        raise RuntimeError("no runtime variant passes the scenario-collapse guard")
    ranked = sorted(eligible, key=_selection_key)
    winner = ranked[0]
    winner_id = _variant_id(winner)
    heldout_delta = compare_reports(
        baseline,
        winner,
        RUNTIME_COMPARISON_VARIABLES,
        split_name="held_out",
    )["metric_deltas"]
    heldout_scenario_deltas = {
        scenario: round(
            float(_scenarios(winner, "held_out")[scenario]["hit_rate_at_10"])
            - float(_scenarios(baseline, "held_out")[scenario]["hit_rate_at_10"]),
            12,
        )
        for scenario in sorted(_scenarios(baseline, "held_out"))
    }
    confirmation_checks = {
        "overall_hit_rate": (
            float(heldout_delta["hit_rate_at_10"])
            >= -spec.max_heldout_hit_rate_drop
        ),
        "overall_mrr": (
            float(heldout_delta["mrr"]) >= -spec.max_heldout_mrr_drop
        ),
        "overall_technical_score": (
            float(heldout_delta["technical_score"])
            >= -spec.max_heldout_technical_score_drop
        ),
        "scenario_hit_rate": all(
            delta >= -spec.max_heldout_scenario_hit_rate_drop
            for delta in heldout_scenario_deltas.values()
        ),
    }
    confirmation_passed = all(confirmation_checks.values())
    production_selection = (
        winner_id
        if evidence_tier == "public-development" and confirmation_passed
        else spec.baseline_variant_id
        if evidence_tier == "public-development"
        else None
    )
    return {
        "evidence_tier": evidence_tier,
        "selection_basis": "tuning_rank_then_heldout_gate",
        "tuning_ranking_basis": "tuning_only",
        "status": (
            "fixture_mechanics_only"
            if evidence_tier == "fixture"
            else "heldout_confirmed"
            if confirmation_passed
            else "heldout_rejected_tuning_winner"
        ),
        "baseline_variant_id": spec.baseline_variant_id,
        "selected_variant_id": production_selection,
        "provisional_tuning_leader": winner_id,
        "eligible_tuning_ranking": [_variant_id(report) for report in ranked],
        "scenario_collapse_guards": guards,
        "heldout_confirmation": {
            "candidate_variant_id": winner_id,
            "overall_delta_from_baseline": heldout_delta,
            "scenario_hit_rate_deltas": heldout_scenario_deltas,
            "checks": confirmation_checks,
            "passed": confirmation_passed,
            "used_to_accept_or_reject_tuning_winner": (
                evidence_tier == "public-development"
            ),
            "used_to_choose_an_alternative": False,
            "used_for_tuning_ranking": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--expected-count", type=int, default=50_000)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedder-factory")
    parser.add_argument(
        "--evidence-tier",
        choices=("fixture", "public-development"),
        default="public-development",
    )
    parser.add_argument("--split-version", default="public-v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tuning-fraction", type=float, default=0.7)
    arguments = parser.parse_args()

    output_directory = arguments.output_directory.resolve()
    if output_directory.is_relative_to(PROJECT_ROOT):
        parser.error("public M4 output must be outside the source repository")

    revision, dirty = _revision()
    if dirty and arguments.evidence_tier == "public-development":
        parser.error("public M4 evidence requires a clean committed worktree")
    spec = load_retrieval_sweep_spec(arguments.spec)
    if (arguments.artifact is None) != (arguments.embedder_factory is None):
        parser.error("--artifact and --embedder-factory must be supplied together")
    catalog = load_catalog(arguments.catalog, expected_count=arguments.expected_count)
    query_embedder = (
        None
        if arguments.embedder_factory is None
        else _load_embedder(arguments.embedder_factory)
    )
    dense_index = (
        None
        if arguments.artifact is None
        else load_dense_index(
            arguments.artifact,
            catalog,
            embedding_route_id=str(getattr(query_embedder, "route_id", "")),
        )
    )
    if dense_index is not None and arguments.evidence_tier == "public-development":
        identity = " ".join(
            (
                dense_index.manifest.provider,
                dense_index.manifest.model,
                dense_index.manifest.route_id,
            )
        ).casefold()
        if any(
            marker in identity
            for marker in ("fixture", "fake", "test-only", "semantic-keywords")
        ):
            parser.error(
                "fixture embedding artifacts cannot support public-development selection"
            )
    variants = (
        spec.variants
        if dense_index is not None
        else tuple(variant for variant in spec.variants if not variant.requires_dense)
    )
    if spec.baseline_variant_id not in {variant.variant_id for variant in variants}:
        parser.error("the M4 baseline must be executable")
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
    route_execution: dict[str, object] = {}
    embedding_usage: dict[str, object] = {}
    output_directory.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        config = _experiment_config(
            variant,
            spec,
            split_version=arguments.split_version,
            seed=arguments.seed,
            catalog_checksum=catalog_checksum,
            code_revision=revision,
            dense_index=dense_index if variant.requires_dense else None,
        )
        created_agents: list[ShoppingAgent[SessionState]] = []
        trackers: list[_RouteTracker] = []
        _take_embedding_usage(query_embedder)

        def factory(selected=variant):
            tracker = _RouteTracker()
            agent = _build_agent(
                catalog,
                selected,
                dense_index=dense_index,
                query_embedder=query_embedder,
                route_tracker=tracker,
            )
            created_agents.append(agent)
            trackers.append(tracker)
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
        expected_route = (
            "hybrid" if variant.route == "auto" else variant.route
        )
        route_execution[variant.variant_id] = _validated_route_execution(
            variant.variant_id,
            expected_route,
            trackers,
        )
        usage = _take_embedding_usage(query_embedder)
        embedding_usage[variant.variant_id] = (
            None if usage is None else embedding_usage_as_dict(usage)
        )
        reports.append(report)
        path = output_directory / f"{variant.variant_id}.json"
        path.write_text(canonical_report_json(report), encoding="utf-8")

    baseline = next(
        report for report in reports if _variant_id(report) == spec.baseline_variant_id
    )
    comparisons = {
        _variant_id(report): {
            "tuning": compare_reports(
                baseline,
                report,
                RUNTIME_COMPARISON_VARIABLES,
                split_name="tuning",
            ),
            "held_out": compare_reports(
                baseline,
                report,
                RUNTIME_COMPARISON_VARIABLES,
                split_name="held_out",
            ),
        }
        for report in reports
    }
    executed_variant_ids = [_variant_id(report) for report in reports]
    summary = {
        "runtime_sweep_schema_version": "m4-retrieval-runtime-v2",
        "evidence_tier": arguments.evidence_tier,
        "code_revision": revision,
        "code_dirty": False,
        "catalog_checksum": catalog_checksum,
        "dataset_checksum": _sha256(arguments.dataset),
        "sweep_spec_fingerprint": spec.fingerprint,
        "split": split.to_dict(),
        "executed_variant_ids": executed_variant_ids,
        "deferred_variant_ids": [
            variant.variant_id
            for variant in spec.variants
            if variant.variant_id not in set(executed_variant_ids)
        ],
        "deferred_reason": (
            None
            if dense_index is not None
            else "Dense artifact and matching production embedder were not supplied."
        ),
        "dense_artifact": (
            None
            if dense_index is None
            else {
                "index_id": dense_index.manifest.index_id,
                "route_id": dense_index.manifest.route_id,
                "provider": dense_index.manifest.provider,
                "model": dense_index.manifest.model,
                "dimension": dense_index.manifest.dimension,
                "backend": dense_index.backend,
            }
        ),
        "route_execution": route_execution,
        "embedding_usage_by_variant": embedding_usage,
        "selection": _select(
            spec,
            reports,
            evidence_tier=arguments.evidence_tier,
        ),
        "comparisons_from_baseline": comparisons,
        "report_files": {
            _variant_id(report): f"{_variant_id(report)}.json" for report in reports
        },
    }
    summary_path = output_directory / "summary.json"
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
