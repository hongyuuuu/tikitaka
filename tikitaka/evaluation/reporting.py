"""Canonical, credential-safe experiment report generation."""

from __future__ import annotations

import json
from typing import Mapping

from tikitaka.evaluation.experiment import ExperimentConfig
from tikitaka.evaluation.splits import SplitManifest


_SECRET_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "authorization",
    "password", "secret", "client_secret", "credentials",
}


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            text_key = str(key)
            cleaned[text_key] = "[REDACTED]" if text_key.lower() in _SECRET_KEYS else _sanitize(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def build_report(
    config: ExperimentConfig,
    split: SplitManifest,
    tuning_result: Mapping[str, object],
    held_out_result: Mapping[str, object],
) -> dict:
    if config.split_version != split.spec.version:
        raise ValueError("experiment and split versions do not match")
    report = {
        "report_schema_version": "1.0.0",
        "experiment": {
            "fingerprint": config.fingerprint,
            "configuration": config.to_dict(),
        },
        "split": split.to_dict(),
        "results": {
            "tuning": dict(tuning_result),
            "held_out": dict(held_out_result),
        },
    }
    return _sanitize(report)  # type: ignore[return-value]


def canonical_report_json(report: Mapping[str, object]) -> str:
    """Serialize with stable ordering and no wall-clock field."""

    return json.dumps(_sanitize(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def normalized_metrics_json(report: Mapping[str, object]) -> str:
    """Byte-stable score evidence, intentionally excluding measured latency."""

    results = report.get("results", {})
    normalized: dict[str, object] = {}
    if isinstance(results, Mapping):
        for split_name in ("tuning", "held_out"):
            result = results.get(split_name, {})
            if isinstance(result, Mapping):
                normalized[split_name] = {
                    "metrics": result.get("metrics"),
                    "scenario_metrics": result.get("scenario_metrics"),
                    "questions": result.get("questions"),
                    "sessions": result.get("sessions"),
                }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
