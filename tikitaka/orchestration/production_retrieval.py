"""Production hybrid retrieval selection with a safe sparse fallback."""

from __future__ import annotations

import os
import ssl
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from tikitaka.retrieval import (
    HybridConfig,
    HybridRetriever,
    load_catalog,
    load_dense_index,
    openai_embedder_from_env,
)
from tikitaka.state.query_builder import QueryBuilderConfig


DENSE_ARTIFACT_ENV = "TIKITAKA_DENSE_ARTIFACT"
PRODUCTION_HYBRID_CONFIG = HybridConfig(
    sparse_weight=1.0,
    dense_weight=0.5,
)


@dataclass(frozen=True)
class ProductionRetrievalSelection:
    retriever: object | None
    query_builder: QueryBuilderConfig
    route_id: str
    failure_code: str | None = None


def _ca_bundle_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    try:
        import certifi

        candidates.append(Path(certifi.where()))
    except (ImportError, OSError):
        pass
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile:
        candidates.append(Path(defaults.cafile))
    candidates.extend(
        (
            Path("/etc/ssl/cert.pem"),
            Path("/etc/ssl/certs/ca-certificates.crt"),
            Path("/opt/homebrew/etc/openssl@3/cert.pem"),
            Path("/usr/local/etc/openssl@3/cert.pem"),
        )
    )
    return tuple(dict.fromkeys(candidates))


def ensure_trusted_ca(environ: Mapping[str, str]) -> str | None:
    """Select an installed CA bundle without disabling TLS verification."""

    configured = environ.get("SSL_CERT_FILE", "").strip()
    if configured:
        return configured
    selected = next((path for path in _ca_bundle_candidates() if path.is_file()), None)
    if selected is None:
        return None
    if isinstance(environ, MutableMapping):
        environ["SSL_CERT_FILE"] = str(selected)
    if environ is os.environ:
        os.environ["SSL_CERT_FILE"] = str(selected)
    return str(selected)


def select_production_retrieval(
    catalog_path: str | Path,
    *,
    profile_weight: float,
    environ: Mapping[str, str] | None = None,
) -> ProductionRetrievalSelection:
    """Build the frozen hybrid route when its external artifact is configured."""

    selected_environ = os.environ if environ is None else environ
    sparse_query = QueryBuilderConfig(
        profile_weight=profile_weight,
        route_policy="sparse",
    )
    artifact = selected_environ.get(DENSE_ARTIFACT_ENV, "").strip()
    if not artifact:
        return ProductionRetrievalSelection(
            retriever=None,
            query_builder=sparse_query,
            route_id="sparse",
            failure_code="dense_artifact_unconfigured",
        )

    ensure_trusted_ca(selected_environ)
    try:
        catalog = load_catalog(catalog_path)
        embedder = openai_embedder_from_env(selected_environ)
        dense = load_dense_index(
            artifact,
            catalog,
            embedding_route_id=embedder.route_id,
        )
        retriever = HybridRetriever(
            catalog,
            dense_index=dense,
            query_embedder=embedder,
            config=PRODUCTION_HYBRID_CONFIG,
        )
    except Exception:
        return ProductionRetrievalSelection(
            retriever=None,
            query_builder=sparse_query,
            route_id="sparse_fallback",
            failure_code="dense_runtime_unavailable",
        )

    return ProductionRetrievalSelection(
        retriever=retriever,
        query_builder=QueryBuilderConfig(
            profile_weight=profile_weight,
            route_policy="hybrid",
            embedding_route_id=dense.manifest.route_id,
            index_id=dense.manifest.index_id,
        ),
        route_id=(
            f"hybrid/sparse-{PRODUCTION_HYBRID_CONFIG.sparse_weight:g}"
            f"/dense-{PRODUCTION_HYBRID_CONFIG.dense_weight:g}"
        ),
    )


__all__ = [
    "DENSE_ARTIFACT_ENV",
    "PRODUCTION_HYBRID_CONFIG",
    "ProductionRetrievalSelection",
    "ensure_trusted_ca",
    "select_production_retrieval",
]
