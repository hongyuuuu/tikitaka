"""Deterministic, fail-closed M6 participant-package construction and audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from tikitaka.config import CONTRACT_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.retrieval.manifests import (
    DENSE_ARTIFACT_FORMAT,
    DENSE_NORMALIZED,
    DENSE_VECTOR_DTYPE,
)
from tikitaka.retrieval.openai_embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    PRODUCTION_EMBEDDING_DIMENSIONS,
)
from tikitaka.retrieval.retriever import RetrievalConfig
from tikitaka.retrieval.sparse import SparseIndexConfig
from tikitaka.retrieval.text import (
    DENSE_QUERY_SCHEMA_VERSION,
    PRODUCT_TEXT_SCHEMA_VERSION,
)
from tikitaka.orchestration.production_retrieval import (
    DENSE_ARTIFACT_ENV,
    PRODUCTION_HYBRID_CONFIG,
)


ROOT = Path(__file__).resolve().parents[1]
DENSE_HANDOFF_REPORT = ROOT / "reports" / "p6-production-index-handoff.json"
HYBRID_SELECTION_REPORT = ROOT / "reports" / "p6-hybrid-selection.json"
SCHEMA_VERSION = "m6-submission-manifest-v1"
MAX_PACKAGE_BYTES = 10 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
TEMPLATE_MAP = {
    ROOT / "submission" / "agent.py": PurePosixPath("agent.py"),
    ROOT / "submission" / "README.md": PurePosixPath("README.md"),
    ROOT / "submission" / "REPORT.md": PurePosixPath("REPORT.md"),
    ROOT / "submission" / "requirements.txt": PurePosixPath("requirements.txt"),
}
SOURCE_DIRECTORIES = (ROOT / "starter", ROOT / "tikitaka")
PACKAGE_TOP_LEVEL = frozenset(
    {
        "agent.py",
        "README.md",
        "REPORT.md",
        "requirements.txt",
        "manifest.json",
        "starter",
        "tikitaka",
    }
)
PACKAGE_ROOT_FILES = frozenset(
    {"agent.py", "README.md", "REPORT.md", "requirements.txt", "manifest.json"}
)
PACKAGE_SOURCE_DIRECTORIES = frozenset({"starter", "tikitaka"})
SECRET_PATTERNS = (
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


class ReleaseAuditError(RuntimeError):
    """Raised when a participant package or repository violates M6 policy."""


@dataclass(frozen=True)
class PackageFile:
    source: Path
    archive_path: PurePosixPath
    size: int
    sha256: str


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_identity() -> tuple[str, bool]:
    return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))


def audit_tracked_repository() -> dict[str, object]:
    """Reject committed catalogs, secrets, transient output, and large artifacts."""

    tracked = tuple(line for line in _git("ls-files").splitlines() if line)
    violations: list[str] = []
    oversized: list[dict[str, object]] = []
    forbidden_exact = {
        ".env",
        ".DS_Store",
        "data/catalog.jsonl",
        "results.json",
    }
    forbidden_names = {
        "vectors.f32",
        "build.checkpoint.json",
        "id_rsa",
        "id_ed25519",
    }
    for relative in tracked:
        path = ROOT / relative
        pure = PurePosixPath(relative)
        if (
            relative in forbidden_exact
            or pure.name in forbidden_names
            or pure.suffix in {".pyc", ".log", ".pem", ".key"}
            or "__pycache__" in pure.parts
        ):
            violations.append(relative)
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            oversized.append({"path": relative, "bytes": path.stat().st_size})
    if violations or oversized:
        details = []
        if violations:
            details.append("forbidden=" + ",".join(sorted(violations)))
        if oversized:
            details.append(
                "oversized=" + ",".join(str(item["path"]) for item in oversized)
            )
        raise ReleaseAuditError("tracked repository policy failed: " + " ".join(details))
    return {
        "tracked_file_count": len(tracked),
        "forbidden_tracked_files": [],
        "oversized_tracked_files": [],
        "max_tracked_file_bytes": MAX_FILE_BYTES,
    }


def collect_package_files() -> tuple[PackageFile, ...]:
    sources: list[tuple[Path, PurePosixPath]] = list(TEMPLATE_MAP.items())
    for directory in SOURCE_DIRECTORIES:
        for source in sorted(directory.rglob("*.py")):
            if "__pycache__" not in source.parts:
                sources.append((source, PurePosixPath(source.relative_to(ROOT).as_posix())))
    package_files: list[PackageFile] = []
    seen: set[PurePosixPath] = set()
    for source, archive_path in sorted(sources, key=lambda item: str(item[1])):
        if archive_path in seen:
            raise ReleaseAuditError(f"duplicate package path: {archive_path}")
        seen.add(archive_path)
        if source.is_symlink() or not source.is_file():
            raise ReleaseAuditError(f"package source is not a regular file: {source}")
        size = source.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ReleaseAuditError(f"package source exceeds file limit: {source}")
        payload = source.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                raise ReleaseAuditError(f"possible secret in package source: {source}")
        package_files.append(PackageFile(source, archive_path, size, sha256_path(source)))
    total = sum(item.size for item in package_files)
    if total > MAX_PACKAGE_BYTES:
        raise ReleaseAuditError(f"package sources exceed {MAX_PACKAGE_BYTES} bytes")
    return tuple(package_files)


def _catalog_identity(catalog: Path) -> dict[str, object]:
    if not catalog.is_file():
        raise ReleaseAuditError(f"catalog not found: {catalog}")
    row_count = 0
    with catalog.open("rb") as handle:
        for line in handle:
            if line.strip():
                row_count += 1
    if row_count != 50_000:
        raise ReleaseAuditError(f"catalog row count must be 50000, found {row_count}")
    return {
        "external_path": "data/catalog.jsonl",
        "included": False,
        "row_count": row_count,
        "sha256": sha256_path(catalog),
    }


def build_manifest(
    files: Iterable[PackageFile],
    *,
    catalog: Path,
    revision: str,
    dirty: bool,
) -> dict[str, object]:
    listed = tuple(files)
    return {
        "schema_version": SCHEMA_VERSION,
        "code_revision": revision,
        "code_dirty": dirty,
        "entry_file": "agent.py",
        "python_requires": ">=3.10",
        "mandatory_dependencies": [],
        "catalog": _catalog_identity(catalog),
        "runtime": {
            "network_required": False,
            "network_optional": True,
            "optional_credential": "OPENAI_API_KEY",
            "optional_dense_artifact_environment": DENSE_ARTIFACT_ENV,
            "primary_generative_route": "openai/gpt-5.6-terra/medium",
            "primary_retrieval_route": "hybrid/sparse-1/dense-0.5",
            "degraded_route": "heuristic/local",
            "degraded_retrieval_route": "sparse",
            "local_generative_llm": False,
        },
        "dense_index": {
            "included": False,
            "status": "validated_external_artifact",
            "provider": "openai",
            "model": DEFAULT_EMBEDDING_MODEL,
            "dimensions": PRODUCTION_EMBEDDING_DIMENSIONS,
            "artifact_format": DENSE_ARTIFACT_FORMAT,
            "vector_dtype": DENSE_VECTOR_DTYPE,
            "normalized": DENSE_NORMALIZED,
        },
        "frozen_versions": {
            "contract": CONTRACT_VERSION,
            "structured_output_schema": STRUCTURED_OUTPUT_SCHEMA_VERSION,
            "product_text_schema": PRODUCT_TEXT_SCHEMA_VERSION,
            "dense_query_schema": DENSE_QUERY_SCHEMA_VERSION,
        },
        "frozen_retrieval_config": {
            "sparse_engine": "sqlite-fts5-bm25-v1",
            "sparse_tokenizer": "unicode61 remove_diacritics 2",
            "sparse": asdict(SparseIndexConfig()),
            "structured_ranking": asdict(RetrievalConfig()),
            "hybrid": asdict(PRODUCTION_HYBRID_CONFIG),
        },
        "package_policy": {
            "max_file_bytes": MAX_FILE_BYTES,
            "max_package_bytes": MAX_PACKAGE_BYTES,
            "catalog_included": False,
            "evaluator_included": False,
            "public_labels_included": False,
            "generated_index_included": False,
        },
        "files": {
            str(item.archive_path): {"bytes": item.size, "sha256": item.sha256}
            for item in listed
        },
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build_submission(
    output: str | Path,
    *,
    catalog: str | Path = ROOT / "data" / "catalog.jsonl",
    allow_dirty: bool = False,
) -> dict[str, object]:
    output_path = Path(output).resolve()
    catalog_path = Path(catalog).resolve()
    if output_path.is_relative_to(ROOT):
        raise ReleaseAuditError("submission archive must be written outside the repository")
    if output_path.suffix.casefold() != ".zip":
        raise ReleaseAuditError("submission archive output must use a .zip suffix")
    if output_path == catalog_path:
        raise ReleaseAuditError("submission output must not overwrite the catalog")
    audit_tracked_repository()
    revision, dirty = git_identity()
    if dirty and not allow_dirty:
        raise ReleaseAuditError("submission build requires a clean committed worktree")
    files = collect_package_files()
    manifest = build_manifest(
        files,
        catalog=catalog_path,
        revision=revision,
        dirty=dirty,
    )
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for item in files:
            archive.writestr(_zip_info(str(item.archive_path)), item.source.read_bytes())
        archive.writestr(_zip_info("manifest.json"), manifest_bytes)
    audit_submission_archive(output_path)
    return {
        "archive": str(output_path),
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": sha256_path(output_path),
        "manifest": manifest,
    }


def audit_submission_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise ReleaseAuditError(f"submission archive not found: {archive_path}")
    if archive_path.stat().st_size > MAX_PACKAGE_BYTES:
        raise ReleaseAuditError("submission archive exceeds package size limit")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ReleaseAuditError("submission archive contains duplicate paths")
        if "manifest.json" not in names or "agent.py" not in names:
            raise ReleaseAuditError("submission archive lacks entry file or manifest")
        total = 0
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ReleaseAuditError(f"unsafe archive path: {info.filename}")
            if pure.parts[0] not in PACKAGE_TOP_LEVEL:
                raise ReleaseAuditError(f"disallowed package path: {info.filename}")
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ReleaseAuditError(f"symlink is forbidden in package: {info.filename}")
            if pure.parts[0] in PACKAGE_ROOT_FILES and len(pure.parts) != 1:
                raise ReleaseAuditError(f"root file used as directory: {info.filename}")
            if pure.parts[0] in PACKAGE_SOURCE_DIRECTORIES:
                if len(pure.parts) < 2 or pure.suffix != ".py":
                    raise ReleaseAuditError(f"non-Python source path: {info.filename}")
            if info.file_size > MAX_FILE_BYTES:
                raise ReleaseAuditError(f"oversized package file: {info.filename}")
            total += info.file_size
            payload = archive.read(info)
            for pattern in SECRET_PATTERNS:
                if pattern.search(payload):
                    raise ReleaseAuditError(f"possible secret in archive: {info.filename}")
        if total > MAX_PACKAGE_BYTES:
            raise ReleaseAuditError("uncompressed submission exceeds package size limit")
        manifest = json.loads(archive.read("manifest.json"))
        expected_files = manifest.get("files")
        if not isinstance(expected_files, dict):
            raise ReleaseAuditError("manifest files field is invalid")
        actual = {name for name in names if name != "manifest.json"}
        if actual != set(expected_files):
            raise ReleaseAuditError("archive contents do not match manifest")
        for name, record in expected_files.items():
            if not isinstance(record, dict):
                raise ReleaseAuditError(f"invalid manifest file record: {name}")
            payload = archive.read(name)
            digest = hashlib.sha256(payload).hexdigest()
            if record.get("bytes") != len(payload) or record.get("sha256") != digest:
                raise ReleaseAuditError(f"manifest mismatch for {name}")
    return {
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_path(archive_path),
        "file_count": len(names),
        "uncompressed_bytes": total,
        "forbidden_contents": [],
    }


def _describe_path(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise (tests, odd roots)."""

    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def production_dense_measurements(
    report_path: Path = DENSE_HANDOFF_REPORT,
    selection_report_path: Path | None = None,
) -> dict[str, object]:
    """Report the dense-index position from evidence rather than assertion.

    This block used to hardcode ``pending_production_1024_index`` and list the
    measurements it was waiting for. That was true when it was written and
    silently false afterwards: the artifact was built, measured, and rejected,
    while every audit generated since kept announcing the work as outstanding.

    Reading Person 2's handoff report keeps the field honest in both
    directions. A missing or malformed report means we genuinely have no
    evidence, and pending is then the correct answer rather than a stale one.
    """

    pending: dict[str, object] = {
        "status": "pending_production_1024_index",
        "evidence": None,
        "measurements": {
            key: None
            for key in (
                "index_bytes",
                "build_time_ms",
                "query_time_ms",
                "embedding_cost_usd",
                "production_hybrid_quality_delta",
            )
        },
    }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pending

    artifact = report.get("artifact") or {}
    build = report.get("build") or {}
    selection = report.get("selection") or {}
    comparison = report.get("tuning_comparison") or {}
    query_usage = comparison.get("hybrid_query_embedding_usage") or {}
    score = (comparison.get("overall") or {}).get("technical_score") or {}
    if not artifact or artifact.get("load_validation") != "passed":
        return pending

    selected = bool(selection.get("production_hybrid_selected"))
    release_policy = selection.get("release_retrieval_policy")
    selected_config: dict[str, object] = {}
    selected_result: dict[str, object] = {}
    selection_reason = selection.get("reason")
    if (
        selection_report_path is None
        and report_path.resolve() == DENSE_HANDOFF_REPORT.resolve()
    ):
        selection_report_path = HYBRID_SELECTION_REPORT
    if selection_report_path is not None:
        try:
            owner_selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner_selection = {}
        candidate_config = owner_selection.get("selected_configuration") or {}
        if (
            owner_selection.get("production_hybrid_selected") is True
            and candidate_config.get("index_id") == artifact.get("index_id")
        ):
            selected = True
            release_policy = owner_selection.get("release_retrieval_policy", "hybrid")
            selection_reason = owner_selection.get("selection_reason")
            selected_config = candidate_config
            selected_result = owner_selection.get("tuning_result") or {}
    seconds = build.get("wall_duration_seconds")
    selected_metrics = selected_result.get("metrics") or {}
    selected_score = selected_metrics.get("technical_score")
    sparse_score = ((comparison.get("overall") or {}).get("technical_score") or {}).get("sparse")
    selected_delta = (
        None
        if selected_score is None or sparse_score is None
        else round(float(selected_score) - float(sparse_score), 6)
    )
    return {
        "status": "built_and_selected" if selected else "built_not_selected",
        "evidence": {
            "report": _describe_path(report_path),
            "report_sha256": sha256_path(report_path),
            "source_revision": report.get("source_revision"),
        },
        "index_id": artifact.get("index_id"),
        "route_id": artifact.get("route_id"),
        "catalog_sha256": artifact.get("catalog_sha256"),
        "selected_for_release": selected,
        "release_retrieval_policy": release_policy,
        "selection_reason": selection_reason,
        "selected_configuration": selected_config or None,
        "selected_tuning_result": selected_result or None,
        "measurements": {
            "index_bytes": artifact.get("total_bytes"),
            "build_time_ms": None if seconds is None else int(seconds * 1000),
            "query_time_ms": query_usage.get("latency_ms"),
            "embedding_cost_usd": build.get("unique_artifact_embedding_cost_usd"),
            "production_hybrid_quality_delta": (
                selected_delta
                if selected_delta is not None
                else score.get("delta_hybrid_minus_sparse")
            ),
        },
    }


__all__ = [
    "DENSE_HANDOFF_REPORT",
    "HYBRID_SELECTION_REPORT",
    "production_dense_measurements",
    "ReleaseAuditError",
    "audit_submission_archive",
    "audit_tracked_repository",
    "build_submission",
    "collect_package_files",
    "git_identity",
    "sha256_path",
]
