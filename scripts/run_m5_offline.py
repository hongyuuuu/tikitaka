#!/usr/bin/env python3
"""Produce reproducible M5 evidence with credentials absent and network denied.

The official evaluator tolerates per-turn exceptions, so a successful evaluator
process alone does not prove an offline route avoided the network. This runner
installs a Python audit hook that records and refuses socket connection/DNS
events, observes every Agent response before evaluator normalization, and fails
the run if either boundary was violated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import load_dense_index_safe


SCHEMA_VERSION = "m5-offline-evidence-v1"
EXPECTED_ROUTE = "heuristic/local"
EXPECTED_SCENARIOS = frozenset({"buying", "browsing", "intent_override", "boundary"})
NETWORK_AUDIT_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.sendto",
    }
)


class OfflineEvidenceError(RuntimeError):
    """Raised when an M5 run does not prove its claimed offline boundary."""


class NetworkAuditGuard:
    """Deny and count Python socket activity while explicitly activated."""

    def __init__(self) -> None:
        self.active = False
        self.attempts: Counter[str] = Counter()

    def __call__(self, event: str, _arguments: object) -> None:
        if self.active and event in NETWORK_AUDIT_EVENTS:
            self.attempts[event] += 1
            raise OfflineEvidenceError(f"network access denied during M5 evidence: {event}")

    @contextmanager
    def deny(self) -> Iterator[None]:
        if self.active:
            raise RuntimeError("network audit guard is already active")
        self.active = True
        try:
            yield
        finally:
            self.active = False


@contextmanager
def credential_absent(name: str = "OPENAI_API_KEY") -> Iterator[None]:
    """Remove one credential for the run and restore the process afterward."""

    existed = name in os.environ
    previous = os.environ.pop(name, None)
    try:
        yield
    finally:
        if existed and previous is not None:
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


class ObservedAgent:
    """Validate raw Agent responses before the evaluator can normalize them."""

    def __init__(self, delegate: Agent, catalog_ids: set[str]) -> None:
        self.delegate = delegate
        self.catalog_ids = frozenset(catalog_ids)
        self.response_count = 0
        self.exceptions: Counter[str] = Counter()
        self.violations: Counter[str] = Counter()

    @property
    def route_id(self) -> str:
        return str(self.delegate.route_id)

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.delegate._shopping_agent, "degraded", False))

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            response = self.delegate.respond(session_id, user_message, turn, top_k)
        except Exception as error:
            self.exceptions[type(error).__name__] += 1
            raise
        self.response_count += 1
        self._observe(response, top_k)
        return response

    def close(self) -> None:
        self.delegate.close()

    def _observe(self, response: object, top_k: int) -> None:
        if not isinstance(response, dict):
            self.violations["response_not_object"] += 1
            return
        if not isinstance(response.get("message"), str):
            self.violations["message_not_string"] += 1
        ask_attribute = response.get("ask_attribute")
        allowed = {
            "category",
            "material",
            "color",
            "size",
            "style",
            "brand",
            "budget",
            "feature",
            "use_case",
            "other",
        }
        if ask_attribute is not None and ask_attribute not in allowed:
            self.violations["invalid_ask_attribute"] += 1
        recommendations = response.get("recommendations")
        if not isinstance(recommendations, list):
            self.violations["recommendations_not_list"] += 1
            return
        if ask_attribute is not None and recommendations:
            self.violations["clarify_with_recommendations"] += 1
        if len(recommendations) > top_k:
            self.violations["recommendation_limit_exceeded"] += 1
        identifiers: list[str] = []
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                self.violations["recommendation_not_object"] += 1
                continue
            parent_asin = recommendation.get("parent_asin")
            if not isinstance(parent_asin, str) or parent_asin not in self.catalog_ids:
                self.violations["invalid_catalog_id"] += 1
                continue
            identifiers.append(parent_asin)
        if len(identifiers) != len(set(identifiers)):
            self.violations["duplicate_catalog_id"] += 1
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self.violations["usage_not_object"] += 1
            return
        for field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                self.violations[f"invalid_{field}"] += 1


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> tuple[str, bool]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def probe_dense_artifact_failures(catalog_path: str | Path) -> dict[str, str]:
    """Exercise nonfatal startup behavior for absent and malformed artifacts."""

    catalog = load_catalog(catalog_path)
    with tempfile.TemporaryDirectory(prefix="tikitaka-m5-missing-") as directory:
        missing = load_dense_index_safe(directory, catalog)
    with tempfile.TemporaryDirectory(prefix="tikitaka-m5-corrupt-") as directory:
        Path(directory, "manifest.json").write_bytes(b"not-json")
        corrupt = load_dense_index_safe(directory, catalog)
    checks = {
        "missing_artifact": str(missing.failure_code),
        "corrupt_artifact": str(corrupt.failure_code),
    }
    expected = {
        "missing_artifact": "dense_artifact_unavailable",
        "corrupt_artifact": "dense_artifact_invalid",
    }
    if checks != expected or missing.index is not None or corrupt.index is not None:
        raise OfflineEvidenceError(f"dense artifact failure probe failed: {checks}")
    return checks


def run_offline_evidence(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    expected_catalog_count: int = 50_000,
    expected_sample_count: int = 200,
) -> dict[str, object]:
    """Run the official public simulator under an active network denial."""

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    if len(catalog_ids) != expected_catalog_count:
        raise OfflineEvidenceError(
            f"catalog count mismatch: expected {expected_catalog_count}, found {len(catalog_ids)}"
        )
    if len(samples) != expected_sample_count:
        raise OfflineEvidenceError(
            f"sample count mismatch: expected {expected_sample_count}, found {len(samples)}"
        )

    guard = NetworkAuditGuard()
    sys.addaudithook(guard)
    with credential_absent(), guard.deny():
        observed = ObservedAgent(Agent(catalog_path), catalog_ids)
        try:
            result = evaluate(observed, samples, catalog_ids, categories, products)
        finally:
            observed.close()

    if guard.attempts:
        raise OfflineEvidenceError(
            "network activity was attempted: " + ", ".join(sorted(guard.attempts))
        )
    if observed.route_id != EXPECTED_ROUTE or not observed.degraded:
        raise OfflineEvidenceError(
            f"offline run selected an unexpected route: {observed.route_id!r}"
        )
    if observed.exceptions:
        raise OfflineEvidenceError(
            "Agent exceptions were hidden by the evaluator: "
            + ", ".join(f"{name}={count}" for name, count in sorted(observed.exceptions.items()))
        )
    if observed.violations:
        raise OfflineEvidenceError(
            "raw Agent response violations: "
            + ", ".join(f"{name}={count}" for name, count in sorted(observed.violations.items()))
        )
    usage = result.get("reported_token_usage")
    if not isinstance(usage, Mapping) or usage.get("total_tokens") != 0:
        raise OfflineEvidenceError("offline run reported non-zero model token usage")
    scenario_metrics = result.get("scenario_metrics")
    if not isinstance(scenario_metrics, Mapping) or set(scenario_metrics) != EXPECTED_SCENARIOS:
        raise OfflineEvidenceError("offline result does not cover all official scenarios")

    revision, dirty = _git_identity()
    if dirty:
        raise OfflineEvidenceError("M5 evidence requires a clean committed worktree")
    metrics = {key: value for key, value in result.items() if key != "sessions"}
    return {
        "schema_version": SCHEMA_VERSION,
        "code_revision": revision,
        "code_dirty": False,
        "catalog_checksum": _sha256(catalog_path),
        "dataset_checksum": _sha256(dataset_path),
        "execution": {
            "credential_forced_absent": True,
            "network_guard": "python-audit-hook/socket-connect-dns-sendto",
            "network_attempt_count": 0,
            "route_id": observed.route_id,
            "degraded": observed.degraded,
            "local_generative_llm": False,
            "raw_response_count": observed.response_count,
            "agent_exception_types": dict(observed.exceptions),
            "response_violations": dict(observed.violations),
        },
        "dense_artifact_failure_checks": probe_dense_artifact_failures(catalog_path),
        "metrics": metrics,
        "quality_delta_from_primary_hybrid": {
            "status": "pending_production_hybrid_evidence",
            "reason": (
                "A real text-embedding-3-large/1024 index and primary API run "
                "do not exist yet; fixture embeddings cannot supply a quality baseline."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--expected-catalog-count", type=int, default=50_000)
    parser.add_argument("--expected-sample-count", type=int, default=200)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run_offline_evidence(
        arguments.catalog,
        arguments.dataset,
        expected_catalog_count=arguments.expected_catalog_count,
        expected_sample_count=arguments.expected_sample_count,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "route_id": report["execution"]["route_id"],
                "network_attempt_count": report["execution"]["network_attempt_count"],
                "metrics": report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
