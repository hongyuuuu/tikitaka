#!/usr/bin/env python3
"""Build and execute the M6 bundle in an isolated organizer-style harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m6_release import (
    ReleaseAuditError,
    production_dense_measurements,
    audit_submission_archive,
    audit_tracked_repository,
    build_submission,
    sha256_path,
)


SCHEMA_VERSION = "m6-clean-reproduction-v2"
EXPECTED_SCENARIOS = frozenset({"buying", "browsing", "intent_override", "boundary"})
RUNNER = r'''from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EVENTS = frozenset({
    "socket.connect",
    "socket.connect_ex",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.sendto",
})
attempts = Counter()

def deny_network(event, _arguments):
    if event in EVENTS:
        attempts[event] += 1
        raise RuntimeError("network access denied in isolated M6 harness: " + event)

sys.addaudithook(deny_network)

import agent as submission_entry
import evaluator.local_evaluator as evaluator
import starter.agent as starter_entry
import tikitaka

instances = []
OriginalAgent = evaluator.Agent

class ProbedAgent(OriginalAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instances.append(self)

evaluator.Agent = ProbedAgent
sys.argv = ["local_evaluator", "--output", "results.json"]
try:
    evaluator.main()
finally:
    route_ids = sorted({str(getattr(item, "route_id", "")) for item in instances})
    for item in instances:
        close = getattr(item, "close", None)
        if callable(close):
            close()
    Path("harness-observation.json").write_text(json.dumps({
        "network_attempts": dict(attempts),
        "route_ids": route_ids,
        "module_files": {
            "agent": str(Path(submission_entry.__file__).resolve()),
            "starter": str(Path(starter_entry.__file__).resolve()),
            "tikitaka": str(Path(tikitaka.__file__).resolve()),
            "evaluator": str(Path(evaluator.__file__).resolve()),
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'''


def _isolated_run(
    archive: Path,
    catalog: Path,
    dataset: Path,
    *,
    timeout_s: float,
) -> tuple[dict[str, object], dict[str, object], str]:
    with tempfile.TemporaryDirectory(prefix="tikitaka-m6-clean-") as directory:
        harness = Path(directory).resolve()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(harness)
        shutil.copytree(ROOT / "evaluator", harness / "evaluator")
        data = harness / "data"
        data.mkdir()
        shutil.copy2(catalog, data / "catalog.jsonl")
        shutil.copy2(dataset, data / "public_set.jsonl")
        (harness / "run_isolated.py").write_text(RUNNER, encoding="utf-8")

        environment = dict(os.environ)
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        started = perf_counter()
        completed = subprocess.run(
            (sys.executable, "-B", "run_isolated.py"),
            cwd=harness,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = (perf_counter() - started) * 1_000.0
        if completed.returncode != 0:
            raise ReleaseAuditError(
                "isolated evaluator failed with exit "
                f"{completed.returncode}: {completed.stderr[-2000:]}"
            )
        results = json.loads((harness / "results.json").read_text(encoding="utf-8"))
        observation = json.loads(
            (harness / "harness-observation.json").read_text(encoding="utf-8")
        )
        for name, module_path in observation["module_files"].items():
            resolved = Path(module_path).resolve()
            if not resolved.is_relative_to(harness):
                raise ReleaseAuditError(
                    f"isolated module {name} resolved outside harness: {resolved}"
                )
        observation["wall_time_ms"] = round(elapsed_ms, 3)
        return results, observation, completed.stdout


def verify_submission(
    *,
    archive: str | Path,
    catalog: str | Path = ROOT / "data" / "catalog.jsonl",
    dataset: str | Path = ROOT / "data" / "public_set.jsonl",
    allow_dirty: bool = False,
    timeout_s: float = 600.0,
) -> dict[str, object]:
    archive_path = Path(archive).resolve()
    catalog_path = Path(catalog).resolve()
    dataset_path = Path(dataset).resolve()
    package = build_submission(
        archive_path,
        catalog=catalog_path,
        allow_dirty=allow_dirty,
    )
    archive_audit = audit_submission_archive(archive_path)
    repository_audit = audit_tracked_repository()
    results, observation, stdout = _isolated_run(
        archive_path,
        catalog_path,
        dataset_path,
        timeout_s=timeout_s,
    )
    sessions = results.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 200:
        raise ReleaseAuditError("isolated evaluator did not produce 200 sessions")
    scenario_metrics = results.get("scenario_metrics")
    if not isinstance(scenario_metrics, dict) or set(scenario_metrics) != EXPECTED_SCENARIOS:
        raise ReleaseAuditError("isolated evaluator did not cover all scenarios")
    usage = results.get("reported_token_usage")
    if not isinstance(usage, dict) or usage.get("total_tokens") != 0:
        raise ReleaseAuditError("offline isolated run reported non-zero model usage")
    if observation.get("network_attempts"):
        raise ReleaseAuditError("isolated run attempted network access")
    if observation.get("route_ids") != ["heuristic/local"]:
        raise ReleaseAuditError(
            f"isolated run selected unexpected routes: {observation.get('route_ids')}"
        )
    manifest = package["manifest"]
    if not isinstance(manifest, dict):
        raise ReleaseAuditError("package manifest is missing")
    metrics = {key: value for key, value in results.items() if key != "sessions"}
    return {
        "schema_version": SCHEMA_VERSION,
        "code_revision": manifest["code_revision"],
        "code_dirty": manifest["code_dirty"],
        "package": {
            "bytes": package["archive_bytes"],
            "sha256": package["archive_sha256"],
            "file_count": archive_audit["file_count"],
            "uncompressed_bytes": archive_audit["uncompressed_bytes"],
            "manifest_schema": manifest["schema_version"],
        },
        "artifact_policy": {
            **repository_audit,
            "archive_forbidden_contents": archive_audit["forbidden_contents"],
            "catalog_in_bundle": False,
            "evaluator_in_bundle": False,
            "public_labels_in_bundle": False,
            "dense_index_in_bundle": False,
            "secret_material_in_bundle": False,
        },
        "clean_reproduction": {
            "isolated_temporary_harness": True,
            "source_repository_on_import_path": False,
            "credential_forced_absent": True,
            "network_guard": "python-audit-hook/socket-connect-dns-sendto",
            "network_attempt_count": 0,
            "route_ids": observation["route_ids"],
            "wall_time_ms": observation["wall_time_ms"],
            "catalog_sha256": sha256_path(catalog_path),
            "dataset_sha256": sha256_path(dataset_path),
            "evaluator_stdout_captured": bool(stdout.strip()),
        },
        "metrics": metrics,
        "production_dense_measurements": production_dense_measurements(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    report = verify_submission(
        archive=arguments.archive,
        catalog=arguments.catalog,
        dataset=arguments.dataset,
        allow_dirty=arguments.allow_dirty,
        timeout_s=arguments.timeout,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
