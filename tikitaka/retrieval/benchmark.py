"""Offline retrieval-only coverage evaluation with explicit split boundaries.

This module never constructs cases from evaluator internals. Person 4 supplies
versioned, precomputed cases containing only an active retrieval request and a
target ID. The target is compared after retrieval and is never passed into a
retriever.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .request import RetrievalConstraint, RetrievalRequest


ALLOWED_SPLITS = frozenset({"tuning", "heldout"})
ALLOWED_SCENARIOS = frozenset({"buying", "browsing", "intent_override", "boundary"})


class BenchmarkValidationError(ValueError):
    """Raised when an offline retrieval case or result violates the contract."""


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    case_id: str
    split: str
    scenario: str
    target_parent_asin: str
    request: RetrievalRequest

    def __post_init__(self) -> None:
        for name in ("case_id", "target_parent_asin"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkValidationError(f"{name} must be a non-empty string")
            if value != value.strip():
                raise BenchmarkValidationError(f"{name} must not contain edge whitespace")
        if not isinstance(self.split, str):
            raise BenchmarkValidationError("split must be a string")
        if not isinstance(self.scenario, str):
            raise BenchmarkValidationError("scenario must be a string")
        if self.split not in ALLOWED_SPLITS:
            raise BenchmarkValidationError(f"unsupported benchmark split: {self.split}")
        if self.scenario not in ALLOWED_SCENARIOS:
            raise BenchmarkValidationError(f"unsupported benchmark scenario: {self.scenario}")
        if not isinstance(self.request, RetrievalRequest):
            raise BenchmarkValidationError("request must be a RetrievalRequest")


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    case_id: str
    split: str
    scenario: str
    target_parent_asin: str
    target_rank: int | None
    returned_count: int
    elapsed_ms: float


def _tuple(payload: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = payload.get(name, ())
    if not isinstance(value, (list, tuple)):
        raise BenchmarkValidationError(f"request.{name} must be an array")
    return tuple(value)


def _string_tuple(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    values = _tuple(payload, name)
    if not all(isinstance(value, str) for value in values):
        raise BenchmarkValidationError(f"request.{name} must contain only strings")
    return tuple(values)


def _request(payload: object) -> RetrievalRequest:
    if not isinstance(payload, Mapping):
        raise BenchmarkValidationError("request must be an object")
    constraints_payload = payload.get("constraints", ())
    if not isinstance(constraints_payload, (list, tuple)):
        raise BenchmarkValidationError("request.constraints must be an array")
    constraints: list[RetrievalConstraint] = []
    for raw in constraints_payload:
        if not isinstance(raw, Mapping):
            raise BenchmarkValidationError("each request constraint must be an object")
        unknown = set(raw).difference(
            {"attribute", "values", "polarity", "strength", "operator", "needs_revalidation"}
        )
        if unknown:
            raise BenchmarkValidationError(
                "unknown request constraint fields: " + ", ".join(sorted(unknown))
            )
        values = raw.get("values", ())
        if not isinstance(values, (list, tuple)):
            raise BenchmarkValidationError("constraint.values must be an array")
        needs_revalidation = raw.get("needs_revalidation", False)
        if not isinstance(needs_revalidation, bool):
            raise BenchmarkValidationError("constraint.needs_revalidation must be a bool")
        try:
            constraints.append(
                RetrievalConstraint(
                    attribute=str(raw["attribute"]),
                    values=tuple(values),
                    polarity=str(raw.get("polarity", "include")),
                    strength=str(raw.get("strength", "soft")),
                    operator=str(raw.get("operator", "eq")),
                    needs_revalidation=needs_revalidation,
                )
            )
        except KeyError as error:
            raise BenchmarkValidationError("constraint.attribute is required") from error
    allowed = {
        "text_query",
        "must_terms",
        "should_terms",
        "exclude_terms",
        "constraints",
        "mode",
        "intent_version",
        "no_preference",
        "profile_terms",
        "profile_weight",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise BenchmarkValidationError(
            "unknown request fields: " + ", ".join(sorted(unknown))
        )
    text_query = payload.get("text_query", "")
    mode = payload.get("mode", "unknown")
    intent_version = payload.get("intent_version", 1)
    profile_weight = payload.get("profile_weight", 0.0)
    if not isinstance(text_query, str):
        raise BenchmarkValidationError("request.text_query must be a string")
    if not isinstance(mode, str):
        raise BenchmarkValidationError("request.mode must be a string")
    if isinstance(intent_version, bool) or not isinstance(intent_version, int):
        raise BenchmarkValidationError("request.intent_version must be an integer")
    if isinstance(profile_weight, bool) or not isinstance(profile_weight, (int, float)):
        raise BenchmarkValidationError("request.profile_weight must be numeric")
    return RetrievalRequest(
        text_query=text_query,
        must_terms=_string_tuple(payload, "must_terms"),
        should_terms=_string_tuple(payload, "should_terms"),
        exclude_terms=_string_tuple(payload, "exclude_terms"),
        constraints=tuple(constraints),
        mode=mode,
        intent_version=intent_version,
        no_preference=frozenset(_string_tuple(payload, "no_preference")),
        profile_terms=_string_tuple(payload, "profile_terms"),
        profile_weight=float(profile_weight),
    )


def load_retrieval_benchmark_cases(
    path: str | Path,
    *,
    valid_ids: frozenset[str],
    require_both_splits: bool = True,
    require_all_scenarios_per_split: bool = True,
) -> tuple[RetrievalBenchmarkCase, ...]:
    """Load strict JSONL cases produced by the integration/evaluation owner."""

    cases: list[RetrievalBenchmarkCase] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise BenchmarkValidationError(
                    f"malformed benchmark JSON at line {line_number}"
                ) from error
            if not isinstance(payload, Mapping):
                raise BenchmarkValidationError(
                    f"benchmark line {line_number} must be an object"
                )
            required = {"case_id", "split", "scenario", "target_parent_asin", "request"}
            missing = required.difference(payload)
            unknown = set(payload).difference(required)
            if missing or unknown:
                details = []
                if missing:
                    details.append("missing=" + ",".join(sorted(missing)))
                if unknown:
                    details.append("unknown=" + ",".join(sorted(unknown)))
                raise BenchmarkValidationError(
                    f"invalid benchmark fields at line {line_number}: " + " ".join(details)
                )
            case = RetrievalBenchmarkCase(
                case_id=payload["case_id"],
                split=payload["split"],
                scenario=payload["scenario"],
                target_parent_asin=payload["target_parent_asin"],
                request=_request(payload["request"]),
            )
            if case.case_id in seen:
                raise BenchmarkValidationError(f"duplicate case_id: {case.case_id}")
            if case.target_parent_asin not in valid_ids:
                raise BenchmarkValidationError(
                    f"target is not in the frozen catalog: {case.target_parent_asin}"
                )
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise BenchmarkValidationError("benchmark case file is empty")
    splits = {case.split for case in cases}
    if require_both_splits and splits != ALLOWED_SPLITS:
        raise BenchmarkValidationError("benchmark must contain tuning and heldout cases")
    targets_by_split = {
        split: {case.target_parent_asin for case in cases if case.split == split}
        for split in splits
    }
    if ALLOWED_SPLITS.issubset(targets_by_split):
        leaked_targets = targets_by_split["tuning"].intersection(
            targets_by_split["heldout"]
        )
        if leaked_targets:
            preview = ", ".join(sorted(leaked_targets)[:5])
            suffix = "" if len(leaked_targets) <= 5 else ", ..."
            raise BenchmarkValidationError(
                "benchmark target leakage across tuning and heldout splits: "
                + preview
                + suffix
            )
    if require_all_scenarios_per_split:
        for split in sorted(splits):
            scenarios = {case.scenario for case in cases if case.split == split}
            missing_scenarios = ALLOWED_SCENARIOS.difference(scenarios)
            if missing_scenarios:
                raise BenchmarkValidationError(
                    f"benchmark split {split} is missing scenarios: "
                    + ", ".join(sorted(missing_scenarios))
                )
    return tuple(cases)


def _result_ids(
    results: Iterable[object],
    *,
    valid_ids: frozenset[str],
    limit: int,
) -> tuple[str, ...]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for result in results:
        parent_asin = getattr(result, "parent_asin", None)
        if not isinstance(parent_asin, str) or not parent_asin:
            raise BenchmarkValidationError("retriever returned an invalid candidate ID")
        if parent_asin not in valid_ids:
            raise BenchmarkValidationError(
                f"retriever returned an ID outside the frozen catalog: {parent_asin}"
            )
        if parent_asin in seen:
            raise BenchmarkValidationError(f"retriever returned a duplicate ID: {parent_asin}")
        seen.add(parent_asin)
        identifiers.append(parent_asin)
        if len(identifiers) >= limit:
            break
    return tuple(identifiers)


def _metrics(observations: Sequence[RetrievalObservation], ks: tuple[int, ...]) -> dict[str, object]:
    count = len(observations)
    ranks = tuple(item.target_rank for item in observations)
    maximum = max(ks)
    hit_ranks = tuple(rank for rank in ranks if rank is not None and rank <= maximum)
    return {
        "case_count": count,
        "hit_rate_at_k": {
            str(k): sum(rank is not None and rank <= k for rank in ranks) / count
            for k in ks
        },
        "mrr_at_k": {
            str(k): sum(1.0 / rank for rank in ranks if rank is not None and rank <= k) / count
            for k in ks
        },
        "mean_rank_on_hit_at_max_k": (
            sum(hit_ranks) / len(hit_ranks) if hit_ranks else None
        ),
        "misses_at_max_k": count - len(hit_ranks),
        "mean_returned_candidates": sum(item.returned_count for item in observations) / count,
        "mean_latency_ms": sum(item.elapsed_ms for item in observations) / count,
    }


def evaluate_retrieval_route(
    cases: Sequence[RetrievalBenchmarkCase],
    search: Callable[[RetrievalRequest, int], Sequence[object]],
    *,
    valid_ids: frozenset[str],
    ks: Sequence[int] = (10, 50, 100, 200),
) -> dict[str, object]:
    """Evaluate one pinned route without ever exposing targets to ``search``."""

    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in normalized_ks):
        raise BenchmarkValidationError("benchmark K values must be positive integers")
    if not cases:
        raise BenchmarkValidationError("cannot evaluate an empty benchmark case set")
    maximum = max(normalized_ks)
    observations: list[RetrievalObservation] = []
    for case in cases:
        started = time.perf_counter()
        raw_results = search(case.request, maximum)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if len(raw_results) > maximum:
            raise BenchmarkValidationError(
                f"retriever returned more than the requested {maximum} candidates"
            )
        identifiers = _result_ids(raw_results, valid_ids=valid_ids, limit=maximum)
        try:
            rank = identifiers.index(case.target_parent_asin) + 1
        except ValueError:
            rank = None
        observations.append(
            RetrievalObservation(
                case_id=case.case_id,
                split=case.split,
                scenario=case.scenario,
                target_parent_asin=case.target_parent_asin,
                target_rank=rank,
                returned_count=len(identifiers),
                elapsed_ms=elapsed_ms,
            )
        )
    report: dict[str, object] = {"ks": normalized_ks, "splits": {}}
    split_report: dict[str, object] = {}
    for split in sorted({case.split for case in cases}):
        selected = [item for item in observations if item.split == split]
        scenarios = {
            scenario: _metrics(
                [item for item in selected if item.scenario == scenario],
                normalized_ks,
            )
            for scenario in sorted({item.scenario for item in selected})
        }
        split_report[split] = {
            "overall": _metrics(selected, normalized_ks),
            "scenarios": scenarios,
        }
    report["splits"] = split_report
    report["observations"] = [
        {
            "case_id": item.case_id,
            "split": item.split,
            "scenario": item.scenario,
            "target_rank": item.target_rank,
            "returned_count": item.returned_count,
            "elapsed_ms": item.elapsed_ms,
        }
        for item in observations
    ]
    return report


__all__ = [
    "ALLOWED_SCENARIOS",
    "ALLOWED_SPLITS",
    "BenchmarkValidationError",
    "RetrievalBenchmarkCase",
    "RetrievalObservation",
    "evaluate_retrieval_route",
    "load_retrieval_benchmark_cases",
]
