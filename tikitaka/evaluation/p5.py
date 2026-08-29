"""Phase P5 experiment arms and held-out selection discipline.

This module describes executable combinations; it does not inspect evaluator
labels or mutate participant code.  Every behavioral switch is included in the
arm fingerprint so two differently wired runs cannot share an identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from tikitaka.decision import phase4_arm


RETRIEVAL_POLICIES = frozenset({"sparse", "dense", "hybrid"})
GENERATIVE_POLICIES = frozenset(
    {"deterministic", "api_always", "api_selective", "api_pinned"}
)


@dataclass(frozen=True)
class P5ExperimentArm:
    """One fully named P5 execution arm."""

    name: str
    retrieval_policy: str = "sparse"
    generative_policy: str = "deterministic"
    decision_arm: str = "adaptive-deterministic"
    profile_weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.retrieval_policy not in RETRIEVAL_POLICIES:
            raise ValueError(
                f"retrieval_policy must be one of {sorted(RETRIEVAL_POLICIES)}"
            )
        if self.generative_policy not in GENERATIVE_POLICIES:
            raise ValueError(
                f"generative_policy must be one of {sorted(GENERATIVE_POLICIES)}"
            )
        phase4_arm(self.decision_arm)
        if self.profile_weight is not None and not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be within [0, 1]")

    @property
    def selected_profile_weight(self) -> float:
        configured = phase4_arm(self.decision_arm).profile_weight
        return configured if self.profile_weight is None else self.profile_weight

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def report_parameters(self) -> tuple[tuple[str, object], ...]:
        decision = phase4_arm(self.decision_arm)
        return (
            ("p5_arm_fingerprint", self.fingerprint),
            ("generative_policy", self.generative_policy),
            ("decision_arm", self.decision_arm),
            ("decision_arm_fingerprint", decision.fingerprint),
        )


def select_release_report(
    reports: Sequence[Mapping[str, object]],
    *,
    maximum_scenario_hit_rate_drop: float = 0.05,
) -> Mapping[str, object]:
    """Select by the official objective order after rejecting scenario collapse.

    The first report is the declared baseline.  All comparisons use held-out
    results only.  Hit Rate@10 dominates MRR, which dominates MTTC.
    """

    if not reports:
        raise ValueError("at least one report is required")
    if not 0.0 <= maximum_scenario_hit_rate_drop <= 1.0:
        raise ValueError("maximum_scenario_hit_rate_drop must be within [0, 1]")
    baseline_scenarios = _scenario_metrics(reports[0])
    eligible: list[Mapping[str, object]] = []
    for report in reports:
        scenarios = _scenario_metrics(report)
        if set(scenarios) != set(baseline_scenarios):
            raise ValueError("reports must contain the same held-out scenarios")
        collapsed = any(
            _number(scenarios[name], "hit_rate_at_10")
            < _number(baseline_scenarios[name], "hit_rate_at_10")
            - maximum_scenario_hit_rate_drop
            for name in baseline_scenarios
        )
        if not collapsed:
            eligible.append(report)
    if not eligible:
        raise ValueError("every candidate causes a material held-out scenario collapse")
    return max(eligible, key=_objective_key)


def _held_out(report: Mapping[str, object]) -> Mapping[str, object]:
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("report is missing results")
    held_out = results.get("held_out")
    if not isinstance(held_out, Mapping):
        raise ValueError("report is missing held_out results")
    return held_out


def _scenario_metrics(report: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    scenarios = _held_out(report).get("scenario_metrics")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("report is missing held-out scenario metrics")
    if not all(isinstance(value, Mapping) for value in scenarios.values()):
        raise ValueError("held-out scenario metrics are malformed")
    return scenarios  # type: ignore[return-value]


def _number(values: Mapping[str, object], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name} must be numeric")
    return float(value)


def _objective_key(report: Mapping[str, object]) -> tuple[float, float, float]:
    metrics = _held_out(report).get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("report is missing held-out metrics")
    return (
        _number(metrics, "hit_rate_at_10"),
        _number(metrics, "mrr"),
        -_number(metrics, "mttc"),
    )


__all__ = [
    "GENERATIVE_POLICIES",
    "RETRIEVAL_POLICIES",
    "P5ExperimentArm",
    "select_release_report",
]
