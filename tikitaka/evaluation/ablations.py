"""Compatibility guards and metric deltas for controlled comparisons."""

from __future__ import annotations

from typing import Mapping, Sequence


IDENTITY_FIELDS = {"catalog_checksum", "split_version", "index_id"}


def compare_reports(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    declared_variables: Sequence[str],
    split_name: str = "held_out",
) -> dict:
    base_config = _configuration(baseline)
    candidate_config = _configuration(candidate)
    declared = set(declared_variables)
    unknown = declared - set(base_config)
    if unknown:
        raise ValueError(f"unknown declared experiment variables: {sorted(unknown)}")
    differences = {
        name for name in base_config
        if base_config.get(name) != candidate_config.get(name)
    }
    undeclared = differences - declared - {"name"}
    if undeclared:
        raise ValueError(f"comparison has undeclared configuration differences: {sorted(undeclared)}")
    identity_differences = differences & IDENTITY_FIELDS
    if identity_differences - declared:
        raise ValueError(f"comparison changes experiment identity: {sorted(identity_differences)}")

    base_metrics = _metrics(baseline, split_name)
    candidate_metrics = _metrics(candidate, split_name)
    names = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")
    return {
        "split": split_name,
        "declared_variables": sorted(declared),
        "configuration_differences": sorted(differences),
        "metric_deltas": {
            name: round(float(candidate_metrics[name]) - float(base_metrics[name]), 6)
            for name in names
        },
    }


def _configuration(report: Mapping[str, object]) -> Mapping[str, object]:
    experiment = report.get("experiment")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("configuration"), Mapping):
        raise ValueError("report is missing experiment configuration")
    return experiment["configuration"]  # type: ignore[return-value]


def _metrics(report: Mapping[str, object], split_name: str) -> Mapping[str, object]:
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("report is missing results")
    split = results.get(split_name)
    if not isinstance(split, Mapping) or not isinstance(split.get("metrics"), Mapping):
        raise ValueError(f"report is missing {split_name} metrics")
    return split["metrics"]  # type: ignore[return-value]
