"""Strict serialization and compatibility checks for canonical index manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from types import MappingProxyType

from tikitaka.contracts import IndexManifest

from .catalog import ProductCatalog
from .text import PRODUCT_TEXT_SCHEMA_VERSION


DENSE_ARTIFACT_FORMAT = "dense-f32-v1"
DENSE_VECTOR_DTYPE = "float32-le"
DENSE_NORMALIZED = True
DENSE_IDS_FILENAME = "ids.jsonl"
DENSE_VECTORS_FILENAME = "vectors.f32"
DENSE_MANIFEST_FILENAME = "manifest.json"
DENSE_CHECKPOINT_FILENAME = "build.checkpoint.json"


class ManifestValidationError(ValueError):
    """Raised when the canonical index manifest is malformed or incompatible."""


_MANIFEST_FIELDS = {
    "index_id",
    "catalog_checksum",
    "catalog_row_count",
    "ordered_id_checksum",
    "product_text_schema_version",
    "provider",
    "model",
    "route_id",
    "dimension",
    "vector_dtype",
    "normalized",
    "document_count",
    "artifact_format",
    "built_at",
    "artifact_checksums",
}


def _sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ManifestValidationError(f"{field_name} must be a lowercase SHA-256")


def dense_manifest_as_dict(manifest: IndexManifest) -> dict[str, object]:
    validate_dense_manifest(manifest)
    return {
        "index_id": manifest.index_id,
        "catalog_checksum": manifest.catalog_checksum,
        "catalog_row_count": manifest.catalog_row_count,
        "ordered_id_checksum": manifest.ordered_id_checksum,
        "product_text_schema_version": manifest.product_text_schema_version,
        "provider": manifest.provider,
        "model": manifest.model,
        "route_id": manifest.route_id,
        "dimension": manifest.dimension,
        "vector_dtype": manifest.vector_dtype,
        "normalized": manifest.normalized,
        "document_count": manifest.document_count,
        "artifact_format": manifest.artifact_format,
        "built_at": manifest.built_at,
        "artifact_checksums": dict(sorted(manifest.artifact_checksums.items())),
    }


def dense_manifest_from_dict(payload: object) -> IndexManifest:
    if not isinstance(payload, dict):
        raise ManifestValidationError("dense manifest must be a JSON object")
    missing = _MANIFEST_FIELDS.difference(payload)
    unknown = set(payload).difference(_MANIFEST_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise ManifestValidationError("invalid dense manifest fields: " + " ".join(details))
    checksums = payload["artifact_checksums"]
    if not isinstance(checksums, dict):
        raise ManifestValidationError("artifact_checksums must be an object")
    try:
        manifest = IndexManifest(
            index_id=payload["index_id"],
            catalog_checksum=payload["catalog_checksum"],
            catalog_row_count=payload["catalog_row_count"],
            ordered_id_checksum=payload["ordered_id_checksum"],
            product_text_schema_version=payload["product_text_schema_version"],
            provider=payload["provider"],
            model=payload["model"],
            route_id=payload["route_id"],
            dimension=payload["dimension"],
            vector_dtype=payload["vector_dtype"],
            normalized=payload["normalized"],
            document_count=payload["document_count"],
            artifact_format=payload["artifact_format"],
            built_at=payload["built_at"],
            artifact_checksums=MappingProxyType(dict(checksums)),
        )
    except (TypeError, ValueError) as error:
        raise ManifestValidationError("dense manifest has invalid field types") from error
    validate_dense_manifest(manifest)
    return manifest


def validate_dense_manifest(manifest: IndexManifest) -> None:
    if not isinstance(manifest, IndexManifest):
        raise ManifestValidationError("manifest must use the canonical IndexManifest contract")
    if manifest.artifact_format != DENSE_ARTIFACT_FORMAT:
        raise ManifestValidationError(
            f"unsupported dense artifact format: {manifest.artifact_format}"
        )
    if manifest.vector_dtype != DENSE_VECTOR_DTYPE:
        raise ManifestValidationError(f"unsupported vector dtype: {manifest.vector_dtype}")
    if manifest.normalized is not DENSE_NORMALIZED:
        raise ManifestValidationError("dense vectors must be L2 normalized")
    _sha256(manifest.catalog_checksum, "catalog_checksum")
    _sha256(manifest.ordered_id_checksum, "ordered_id_checksum")
    if not manifest.index_id.startswith("dense-") or len(manifest.index_id) != 26:
        raise ManifestValidationError("index_id has an invalid dense-index format")
    try:
        timestamp = datetime.fromisoformat(manifest.built_at)
    except ValueError as error:
        raise ManifestValidationError("built_at must be ISO-8601") from error
    if timestamp.tzinfo is None:
        raise ManifestValidationError("built_at must include a timezone")
    expected_artifacts = {DENSE_IDS_FILENAME, DENSE_VECTORS_FILENAME}
    if set(manifest.artifact_checksums) != expected_artifacts:
        raise ManifestValidationError(
            "artifact_checksums must contain exactly ids.jsonl and vectors.f32"
        )
    for filename, checksum in manifest.artifact_checksums.items():
        if not isinstance(filename, str) or not isinstance(checksum, str):
            raise ManifestValidationError("artifact checksum names and values must be strings")
        _sha256(checksum, f"artifact checksum for {filename}")


def assert_dense_manifest_compatible(
    manifest: IndexManifest,
    catalog: ProductCatalog,
    *,
    embedding_route_id: str | None = None,
    index_id: str | None = None,
) -> None:
    validate_dense_manifest(manifest)
    mismatches: list[str] = []
    if manifest.catalog_checksum != catalog.identity.source_sha256:
        mismatches.append("catalog_checksum")
    if manifest.catalog_row_count != catalog.identity.row_count:
        mismatches.append("catalog_row_count")
    if manifest.ordered_id_checksum != catalog.identity.ordered_parent_asin_sha256:
        mismatches.append("ordered_id_checksum")
    if manifest.product_text_schema_version != PRODUCT_TEXT_SCHEMA_VERSION:
        mismatches.append("product_text_schema_version")
    expected_index_id = dense_index_id(
        catalog,
        embedding_provider=manifest.provider,
        embedding_model=manifest.model,
        embedding_route_id=manifest.route_id,
        embedding_dimension=manifest.dimension,
    )
    if manifest.index_id != expected_index_id:
        mismatches.append("derived_index_id")
    if embedding_route_id is not None and manifest.route_id != embedding_route_id:
        mismatches.append("embedding_route_id")
    if index_id is not None and manifest.index_id != index_id:
        mismatches.append("index_id")
    if mismatches:
        raise ManifestValidationError("dense index is incompatible: " + ", ".join(mismatches))


def dense_index_id(
    catalog: ProductCatalog,
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_route_id: str,
    embedding_dimension: int,
) -> str:
    identity = {
        "artifact_format": DENSE_ARTIFACT_FORMAT,
        "catalog_checksum": catalog.identity.source_sha256,
        "catalog_row_count": catalog.identity.row_count,
        "ordered_id_checksum": catalog.identity.ordered_parent_asin_sha256,
        "product_text_schema_version": PRODUCT_TEXT_SCHEMA_VERSION,
        "provider": embedding_provider,
        "model": embedding_model,
        "route_id": embedding_route_id,
        "dimension": embedding_dimension,
        "vector_dtype": DENSE_VECTOR_DTYPE,
        "normalized": DENSE_NORMALIZED,
        "document_count": len(catalog),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "dense-" + hashlib.sha256(encoded).hexdigest()[:20]
