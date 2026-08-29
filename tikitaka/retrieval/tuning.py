"""Strict M4 retrieval sweeps with tuning-only selection and held-out confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Mapping, Sequence

from tikitaka.contracts import Usage

from .adapters import contract_candidate
from .benchmark import RetrievalBenchmarkCase, evaluate_retrieval_route
from .catalog import ProductCatalog
from .dense import DenseIndex
from .embedding import embedding_usage_as_dict
from .hybrid import HybridConfig, HybridRetriever
from .request import RetrievalRequest
from .retriever import RetrievalConfig, SparseStructuredRetriever
from .sparse import SparseIndexConfig


SWEEP_SCHEMA_VERSION = "retrieval-m4-sweep-v1"
EVIDENCE_TIERS = frozenset({"fixture", "public-development"})
ROUTES = frozenset({"sparse", "dense", "hybrid", "auto"})


class SweepValidationError(ValueError):
    """Raised when an M4 specification or report cannot support a fair comparison."""


class SweepExecutionError(RuntimeError):
    """Raised when a requested route silently degrades or cannot execute."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SweepValidationError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise SweepValidationError(f"{name} must not contain edge whitespace")
    return value


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepValidationError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise SweepValidationError(f"{name} must be within [0.0, 1.0]")
    return converted


def _strict_dataclass_payload(
    payload: object,
    data_type: type,
    name: str,
) -> dict[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise SweepValidationError(f"{name} must be an object")
    allowed = {item.name for item in fields(data_type)}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise SweepValidationError(
            f"unknown {name} fields: " + ", ".join(sorted(str(item) for item in unknown))
        )
    return {str(key): value for key, value in payload.items()}


@dataclass(frozen=True, slots=True)
class RetrievalVariant:
    variant_id: str
    route: str
    sparse: SparseIndexConfig = SparseIndexConfig()
    ranking: RetrievalConfig = RetrievalConfig()
    hybrid: HybridConfig = HybridConfig()
    request_profile_weight: float | None = None
    candidate_limit: int = 100
    description: str = ""

    def __post_init__(self) -> None:
        _required_text(self.variant_id, "variant_id")
        if self.route not in ROUTES:
            raise SweepValidationError(f"unsupported retrieval route: {self.route}")
        if not isinstance(self.sparse, SparseIndexConfig):
            raise TypeError("sparse must be SparseIndexConfig")
        if not isinstance(self.ranking, RetrievalConfig):
            raise TypeError("ranking must be RetrievalConfig")
        if not isinstance(self.hybrid, HybridConfig):
            raise TypeError("hybrid must be HybridConfig")
        if self.hybrid.ranking != self.ranking:
            raise SweepValidationError("hybrid.ranking must be the variant ranking config")
        if self.request_profile_weight is not None:
            object.__setattr__(
                self,
                "request_profile_weight",
                _unit_interval(self.request_profile_weight, "request_profile_weight"),
            )
        if isinstance(self.candidate_limit, bool) or not isinstance(
            self.candidate_limit, int
        ):
            raise SweepValidationError("candidate_limit must be an integer")
        if self.candidate_limit <= 0:
            raise SweepValidationError("candidate_limit must be positive")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

    @property
    def requires_dense(self) -> bool:
        return self.route in {"dense", "hybrid", "auto"}

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "route": self.route,
            "sparse": asdict(self.sparse),
            "ranking": asdict(self.ranking),
            "hybrid": {
                key: value
                for key, value in asdict(self.hybrid).items()
                if key != "ranking"
            },
            "request_profile_weight": self.request_profile_weight,
            "candidate_limit": self.candidate_limit,
            "description": self.description,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievalSweepSpec:
    baseline_variant_id: str
    selection_k: int
    max_scenario_hit_rate_drop: float
    variants: tuple[RetrievalVariant, ...]
    schema_version: str = SWEEP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SWEEP_SCHEMA_VERSION:
            raise SweepValidationError(
                f"unsupported sweep schema version: {self.schema_version}"
            )
        _required_text(self.baseline_variant_id, "baseline_variant_id")
        if isinstance(self.selection_k, bool) or not isinstance(self.selection_k, int):
            raise SweepValidationError("selection_k must be an integer")
        if self.selection_k <= 0:
            raise SweepValidationError("selection_k must be positive")
        object.__setattr__(
            self,
            "max_scenario_hit_rate_drop",
            _unit_interval(
                self.max_scenario_hit_rate_drop,
                "max_scenario_hit_rate_drop",
            ),
        )
        object.__setattr__(self, "variants", tuple(self.variants))
        if len(self.variants) < 2:
            raise SweepValidationError("a sweep requires at least two variants")
        identifiers = [variant.variant_id for variant in self.variants]
        if len(identifiers) != len(set(identifiers)):
            raise SweepValidationError("sweep variant IDs must be unique")
        if self.baseline_variant_id not in identifiers:
            raise SweepValidationError("baseline_variant_id is absent from variants")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "baseline_variant_id": self.baseline_variant_id,
            "selection_k": self.selection_k,
            "max_scenario_hit_rate_drop": self.max_scenario_hit_rate_drop,
            "variants": [variant.to_dict() for variant in self.variants],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _variant(payload: object) -> RetrievalVariant:
    if not isinstance(payload, Mapping):
        raise SweepValidationError("each sweep variant must be an object")
    allowed = {
        "variant_id",
        "route",
        "sparse",
        "ranking",
        "hybrid",
        "request_profile_weight",
        "candidate_limit",
        "description",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise SweepValidationError(
            "unknown variant fields: " + ", ".join(sorted(str(item) for item in unknown))
        )
    missing = {"variant_id", "route"}.difference(payload)
    if missing:
        raise SweepValidationError(
            "variant is missing fields: " + ", ".join(sorted(missing))
        )
    try:
        sparse = SparseIndexConfig(
            **_strict_dataclass_payload(payload.get("sparse"), SparseIndexConfig, "sparse")
        )
        ranking = RetrievalConfig(
            **_strict_dataclass_payload(payload.get("ranking"), RetrievalConfig, "ranking")
        )
        hybrid_values = _strict_dataclass_payload(
            payload.get("hybrid"), HybridConfig, "hybrid"
        )
        if "ranking" in hybrid_values:
            raise SweepValidationError("variant hybrid config must not redefine ranking")
        hybrid = HybridConfig(**hybrid_values, ranking=ranking)
    except (TypeError, ValueError) as error:
        if isinstance(error, SweepValidationError):
            raise
        raise SweepValidationError(f"invalid variant configuration: {error}") from error
    return RetrievalVariant(
        variant_id=payload["variant_id"],
        route=payload["route"],
        sparse=sparse,
        ranking=ranking,
        hybrid=hybrid,
        request_profile_weight=payload.get("request_profile_weight"),
        candidate_limit=payload.get("candidate_limit", 100),
        description=payload.get("description", ""),
    )


def load_retrieval_sweep_spec(path: str | Path) -> RetrievalSweepSpec:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                SweepValidationError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SweepValidationError(f"cannot read sweep spec: {source}") from error
    if not isinstance(payload, Mapping):
        raise SweepValidationError("sweep spec must be an object")
    allowed = {
        "schema_version",
        "baseline_variant_id",
        "selection_k",
        "max_scenario_hit_rate_drop",
        "variants",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise SweepValidationError(
            "unknown sweep fields: " + ", ".join(sorted(str(item) for item in unknown))
        )
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise SweepValidationError("sweep variants must be an array")
    try:
        return RetrievalSweepSpec(
            schema_version=payload.get("schema_version", ""),
            baseline_variant_id=payload["baseline_variant_id"],
            selection_k=payload["selection_k"],
            max_scenario_hit_rate_drop=payload.get(
                "max_scenario_hit_rate_drop", 0.0
            ),
            variants=tuple(_variant(item) for item in variants),
        )
    except KeyError as error:
        raise SweepValidationError(f"missing sweep field: {error.args[0]}") from error


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_diagnostic_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate_fields = (
        "sparse_candidates",
        "dense_candidates",
        "fused_candidates",
        "hard_filtered_candidates",
        "returned_candidates",
    )
    overlap_depths = sorted(
        {
            int(depth)
            for record in records
            for depth in record.get("route_overlap", {})
        }
    )
    timing_names = sorted(
        {
            str(name)
            for record in records
            for name in record.get("route_timings_ms", {})
        }
    )
    summary: dict[str, object] = {
        "case_count": len(records),
        "mean_candidate_counts": {
            name: _mean([float(record[name]) for record in records])
            for name in candidate_fields
        },
        "mean_route_overlap_count": {
            str(depth): _mean(
                [
                    float(record.get("route_overlap", {}).get(depth, 0))
                    for record in records
                ]
            )
            for depth in overlap_depths
        },
        "mean_route_overlap_rate": {
            str(depth): _mean(
                [
                    (
                        float(record.get("route_overlap", {}).get(depth, 0))
                        / denominator
                        if (
                            denominator := min(
                                depth,
                                int(record["sparse_candidates"]),
                                int(record["dense_candidates"]),
                            )
                        )
                        else 0.0
                    )
                    for record in records
                ]
            )
            for depth in overlap_depths
        },
        "mean_route_timings_ms": {
            name: _mean(
                [
                    float(record.get("route_timings_ms", {}).get(name, 0.0))
                    for record in records
                ]
            )
            for name in timing_names
        },
    }
    for name in (
        "top_score_margin",
        "top_score_concentration",
        "effective_candidate_count",
    ):
        values = [float(record[name]) for record in records if record.get(name) is not None]
        summary["mean_" + name] = _mean(values) if values else None
    return summary


def _diagnostic_report(
    cases: Sequence[RetrievalBenchmarkCase],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(cases) != len(records):
        raise SweepExecutionError("expected exactly one diagnostic record per case")
    indexed = tuple(zip(cases, records))
    splits: dict[str, object] = {}
    for split in sorted({case.split for case in cases}):
        selected = [(case, record) for case, record in indexed if case.split == split]
        scenarios: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for case, record in selected:
            scenarios[case.scenario].append(record)
        splits[split] = {
            "overall": _summarize_diagnostic_records(
                [record for _, record in selected]
            ),
            "scenarios": {
                scenario: _summarize_diagnostic_records(items)
                for scenario, items in sorted(scenarios.items())
            },
        }
    return {"splits": splits}


def _sparse_record(diagnostics: object) -> dict[str, object]:
    return {
        "sparse_candidates": getattr(diagnostics, "sparse_candidates"),
        "dense_candidates": 0,
        "fused_candidates": getattr(diagnostics, "sparse_candidates"),
        "hard_filtered_candidates": getattr(
            diagnostics, "hard_filtered_candidates"
        ),
        "returned_candidates": getattr(diagnostics, "returned_candidates"),
        "route_overlap": {},
        "route_timings_ms": {"total": getattr(diagnostics, "elapsed_ms")},
        "top_score_margin": None,
        "top_score_concentration": None,
        "effective_candidate_count": None,
    }


def _hybrid_record(diagnostics: object) -> dict[str, object]:
    return {
        "sparse_candidates": getattr(diagnostics, "sparse_candidates"),
        "dense_candidates": getattr(diagnostics, "dense_candidates"),
        "fused_candidates": getattr(diagnostics, "fused_candidates"),
        "hard_filtered_candidates": getattr(
            diagnostics, "hard_filtered_candidates"
        ),
        "returned_candidates": getattr(diagnostics, "returned_candidates"),
        "route_overlap": dict(getattr(diagnostics, "route_overlap")),
        "route_timings_ms": dict(getattr(diagnostics, "route_timings_ms")),
        "top_score_margin": getattr(diagnostics, "top_score_margin"),
        "top_score_concentration": getattr(
            diagnostics, "top_score_concentration"
        ),
        "effective_candidate_count": getattr(
            diagnostics, "effective_candidate_count"
        ),
    }


def _take_usage(embedder: object | None) -> Usage | None:
    take = None if embedder is None else getattr(embedder, "take_usage", None)
    if not callable(take):
        return None
    usage = take()
    if not isinstance(usage, Usage):
        raise SweepExecutionError("embedder.take_usage() must return canonical Usage")
    if usage.calls == 0 and not usage.cache_hit:
        return None
    return usage


def _variant_request(
    request: RetrievalRequest,
    variant: RetrievalVariant,
    dense_index: DenseIndex | None,
) -> RetrievalRequest:
    profile_weight = (
        request.profile_weight
        if variant.request_profile_weight is None
        else variant.request_profile_weight
    )
    if variant.requires_dense:
        if dense_index is None:
            raise SweepExecutionError(
                f"variant {variant.variant_id} requires a dense index"
            )
        return replace(
            request,
            profile_weight=profile_weight,
            route_policy=variant.route,
            embedding_route_id=dense_index.manifest.route_id,
            index_id=dense_index.manifest.index_id,
        )
    return replace(
        request,
        profile_weight=profile_weight,
        route_policy="sparse",
        embedding_route_id=None,
        index_id=None,
    )


def run_retrieval_variant(
    variant: RetrievalVariant,
    cases: Sequence[RetrievalBenchmarkCase],
    catalog: ProductCatalog,
    *,
    ks: Sequence[int],
    dense_index: DenseIndex | None = None,
    query_embedder: object | None = None,
    allow_route_degradation: bool = False,
) -> dict[str, object]:
    if variant.requires_dense and (dense_index is None or query_embedder is None):
        raise SweepExecutionError(
            f"variant {variant.variant_id} requires a dense index and query embedder"
        )
    _take_usage(query_embedder)
    records: list[dict[str, object]] = []
    executed: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    if variant.route == "sparse":
        retriever = SparseStructuredRetriever(
            catalog,
            sparse_config=variant.sparse,
            retrieval_config=variant.ranking,
        )
    else:
        retriever = HybridRetriever(
            catalog,
            dense_index=dense_index,
            query_embedder=query_embedder,
            sparse_config=variant.sparse,
            config=variant.hybrid,
        )
    try:
        if variant.route == "sparse":
            def search(request: RetrievalRequest, limit: int):
                selected = _variant_request(request, variant, dense_index)
                result = retriever.retrieve(
                    selected, limit=min(limit, variant.candidate_limit)
                )
                records.append(_sparse_record(result.diagnostics))
                executed["sparse"] += 1
                return [contract_candidate(hit) for hit in result.hits]
        else:
            def search(request: RetrievalRequest, limit: int):
                selected = _variant_request(request, variant, dense_index)
                result = retriever.retrieve(
                    selected, limit=min(limit, variant.candidate_limit)
                )
                records.append(_hybrid_record(result.diagnostics))
                executed[result.diagnostics.executed_route] += 1
                failures.update(result.diagnostics.route_failures)
                return [contract_candidate(hit) for hit in result.hits]

        metrics = evaluate_retrieval_route(
            cases,
            search,
            valid_ids=catalog.ids,
            ks=ks,
        )
    finally:
        retriever.close()
    expected_route = "hybrid" if variant.route == "auto" else variant.route
    degraded = sum(
        count for route, count in executed.items() if route != expected_route
    )
    if (failures or degraded) and not allow_route_degradation:
        details = set(failures)
        details.update(
            "executed_as_" + route for route in executed if route != expected_route
        )
        raise SweepExecutionError(
            f"variant {variant.variant_id} degraded: " + ", ".join(sorted(details))
        )
    usage = _take_usage(query_embedder)
    return {
        "variant_id": variant.variant_id,
        "variant_fingerprint": variant.fingerprint,
        "configuration": variant.to_dict(),
        "metrics": metrics,
        "retrieval_diagnostics": _diagnostic_report(cases, records),
        "execution": {
            "executed_routes": dict(sorted(executed.items())),
            "failure_codes": dict(sorted(failures.items())),
            "degraded_case_count": degraded,
        },
        "embedding_usage": (
            None if usage is None else embedding_usage_as_dict(usage)
        ),
        "ablation_coverage": {
            "case_count": len(cases),
            "profile_term_case_count": sum(bool(case.request.profile_terms) for case in cases),
            "profile_eligible_case_count": sum(
                bool(case.request.profile_terms) and not case.request.constraints
                for case in cases
            ),
        },
    }


def _overall(variant: Mapping[str, object], split: str) -> Mapping[str, object]:
    metrics = variant.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SweepValidationError("variant report is missing metrics")
    splits = metrics.get("splits")
    if not isinstance(splits, Mapping):
        raise SweepValidationError("variant report is missing splits")
    selected = splits.get(split)
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("overall"), Mapping
    ):
        raise SweepValidationError(f"variant report is missing {split} metrics")
    return selected["overall"]  # type: ignore[return-value]


def _scenario_metrics(
    variant: Mapping[str, object], split: str
) -> Mapping[str, Mapping[str, object]]:
    metrics = variant["metrics"]
    splits = metrics["splits"]
    scenarios = splits[split]["scenarios"]
    if not isinstance(scenarios, Mapping):
        raise SweepValidationError("variant report is missing scenario metrics")
    return scenarios  # type: ignore[return-value]


def _at_k(metrics: Mapping[str, object], name: str, k: int) -> float:
    values = metrics.get(name)
    if not isinstance(values, Mapping) or str(k) not in values:
        raise SweepValidationError(f"metrics are missing {name}@{k}")
    return float(values[str(k)])


def _selection_key(variant: Mapping[str, object], k: int) -> tuple[object, ...]:
    metrics = _overall(variant, "tuning")
    ranks = metrics.get("mean_rank_on_hit_at_k")
    if not isinstance(ranks, Mapping):
        raise SweepValidationError("metrics are missing mean_rank_on_hit_at_k")
    mean_rank = ranks.get(str(k))
    normalized_rank = math.inf if mean_rank is None else float(mean_rank)
    return (
        -_at_k(metrics, "hit_rate_at_k", k),
        -_at_k(metrics, "mrr_at_k", k),
        normalized_rank,
        float(metrics["mean_latency_ms"]),
        str(variant["variant_id"]),
    )


def _metric_delta(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    split: str,
    k: int,
) -> dict[str, float | None]:
    base = _overall(baseline, split)
    selected = _overall(candidate, split)
    base_ranks = base.get("mean_rank_on_hit_at_k")
    selected_ranks = selected.get("mean_rank_on_hit_at_k")
    if not isinstance(base_ranks, Mapping) or not isinstance(selected_ranks, Mapping):
        raise SweepValidationError("metrics are missing mean_rank_on_hit_at_k")
    base_rank = base_ranks.get(str(k))
    selected_rank = selected_ranks.get(str(k))
    return {
        "hit_rate_at_k": round(
            _at_k(selected, "hit_rate_at_k", k)
            - _at_k(base, "hit_rate_at_k", k),
            12,
        ),
        "mrr_at_k": round(
            _at_k(selected, "mrr_at_k", k)
            - _at_k(base, "mrr_at_k", k),
            12,
        ),
        "mean_rank_on_hit_at_k": (
            None
            if base_rank is None or selected_rank is None
            else round(float(selected_rank) - float(base_rank), 12)
        ),
        "mean_latency_ms": round(
            float(selected["mean_latency_ms"]) - float(base["mean_latency_ms"]),
            12,
        ),
    }


def select_retrieval_variant(
    spec: RetrievalSweepSpec,
    variant_reports: Sequence[Mapping[str, object]],
    *,
    evidence_tier: str,
) -> dict[str, object]:
    if evidence_tier not in EVIDENCE_TIERS:
        raise SweepValidationError(f"unsupported evidence tier: {evidence_tier}")
    by_id = {str(item.get("variant_id")): item for item in variant_reports}
    expected = {variant.variant_id for variant in spec.variants}
    if set(by_id) != expected:
        raise SweepValidationError("variant reports do not match the sweep specification")
    baseline = by_id[spec.baseline_variant_id]
    baseline_scenarios = _scenario_metrics(baseline, "tuning")
    guards: dict[str, object] = {}
    eligible: list[Mapping[str, object]] = []
    for variant_id, report in sorted(by_id.items()):
        scenarios = _scenario_metrics(report, "tuning")
        missing = set(baseline_scenarios).difference(scenarios)
        if missing:
            raise SweepValidationError(
                f"variant {variant_id} is missing scenarios: "
                + ", ".join(sorted(missing))
            )
        deltas = {
            scenario: round(
                _at_k(scenarios[scenario], "hit_rate_at_k", spec.selection_k)
                - _at_k(
                    baseline_scenarios[scenario],
                    "hit_rate_at_k",
                    spec.selection_k,
                ),
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
        raise SweepValidationError("no variant passes the scenario-collapse guard")
    ranked = sorted(eligible, key=lambda item: _selection_key(item, spec.selection_k))
    provisional = str(ranked[0]["variant_id"])
    selected = provisional if evidence_tier == "public-development" else None
    confirmation_variant = by_id[selected or provisional]
    heldout_scenarios = _scenario_metrics(confirmation_variant, "heldout")
    baseline_heldout_scenarios = _scenario_metrics(baseline, "heldout")
    heldout_confirmation = {
        "variant_id": str(confirmation_variant["variant_id"]),
        "overall_delta_from_baseline": _metric_delta(
            baseline,
            confirmation_variant,
            "heldout",
            spec.selection_k,
        ),
        "scenario_hit_rate_deltas": {
            scenario: round(
                _at_k(
                    heldout_scenarios[scenario],
                    "hit_rate_at_k",
                    spec.selection_k,
                )
                - _at_k(
                    baseline_heldout_scenarios[scenario],
                    "hit_rate_at_k",
                    spec.selection_k,
                ),
                12,
            )
            for scenario in sorted(baseline_heldout_scenarios)
        },
        "used_for_selection": False,
    }
    return {
        "selection_basis": "tuning_only",
        "evidence_tier": evidence_tier,
        "status": (
            "selected" if selected is not None else "fixture_mechanics_only"
        ),
        "baseline_variant_id": spec.baseline_variant_id,
        "selected_variant_id": selected,
        "provisional_tuning_leader": provisional,
        "eligible_tuning_ranking": [str(item["variant_id"]) for item in ranked],
        "scenario_collapse_guards": guards,
        "heldout_confirmation": heldout_confirmation,
    }


def build_retrieval_sweep_report(
    spec: RetrievalSweepSpec,
    variant_reports: Sequence[Mapping[str, object]],
    *,
    evidence_tier: str,
    code_revision: str,
    code_dirty: bool,
    case_file: str,
    case_file_sha256: str,
    catalog: ProductCatalog,
    dense_manifest: Mapping[str, object] | None,
    dense_backend: str | None,
) -> dict[str, object]:
    _required_text(code_revision, "code_revision")
    if not isinstance(code_dirty, bool):
        raise TypeError("code_dirty must be a bool")
    if evidence_tier == "public-development" and code_dirty:
        raise SweepValidationError(
            "public-development evidence requires a clean committed worktree"
        )
    if dense_backend is not None:
        _required_text(dense_backend, "dense_backend")
    selection = select_retrieval_variant(
        spec,
        variant_reports,
        evidence_tier=evidence_tier,
    )
    return {
        "sweep_report_schema_version": "retrieval-m4-report-v1",
        "evidence_tier": evidence_tier,
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "sweep_spec_fingerprint": spec.fingerprint,
        "sweep_specification": spec.to_dict(),
        "case_file": case_file,
        "case_file_sha256": case_file_sha256,
        "catalog_checksum": catalog.identity.source_sha256,
        "catalog_row_count": len(catalog),
        "dense_manifest": None if dense_manifest is None else dict(dense_manifest),
        "dense_backend": dense_backend,
        "variants": list(variant_reports),
        "selection": selection,
        "limitations": (
            [
                "Fixture embeddings and synthetic cases validate mechanics only; "
                "they cannot select a production retrieval configuration."
            ]
            if evidence_tier == "fixture"
            else []
        ),
    }


__all__ = [
    "EVIDENCE_TIERS",
    "RetrievalSweepSpec",
    "RetrievalVariant",
    "SWEEP_SCHEMA_VERSION",
    "SweepExecutionError",
    "SweepValidationError",
    "build_retrieval_sweep_report",
    "load_retrieval_sweep_spec",
    "run_retrieval_variant",
    "select_retrieval_variant",
]
