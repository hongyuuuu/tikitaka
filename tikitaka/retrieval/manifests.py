"""Versioned dense-index identity and compatibility validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from .catalog import ProductCatalog
from .text import PRODUCT_TEXT_SCHEMA_VERSION


DENSE_ARTIFACT_FORMAT_VERSION = "dense-f32-v1"
DENSE_VECTOR_DTYPE = "float32-le"
DENSE_NORMALIZATION = "l2"
DENSE_IDS_FILENAME = "ids.jsonl"
DENSE_VECTORS_FILENAME = "vectors.f32"
DENSE_MANIFEST_FILENAME = "manifest.json"
DENSE_CHECKPOINT_FILENAME = "build.checkpoint.json"


class ManifestValidationError(ValueError):
    """Raised when an index manifest is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    artifact_format_version: str
    index_id: str
    catalog_source_sha256: str
    catalog_row_count: int
    ordered_parent_asin_sha256: str
    product_text_schema_version: str
    embedding_provider: str
    embedding_model: str
    embedding_route_id: str
    embedding_dimension: int
    vector_dtype: str
    normalization: str
    document_count: int
    build_timestamp: str
    artifact_checksums: Mapping[str, str]

    def __post_init__(self) -> None:
        text_fields = (
            "artifact_format_version",
            "index_id",
            "catalog_source_sha256",
            "ordered_parent_asin_sha256",
            "product_text_schema_version",
            "embedding_provider",
            "embedding_model",
            "embedding_route_id",
            "vector_dtype",
            "normalization",
            "build_timestamp",
        )
        for field_name in text_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ManifestValidationError(f"manifest {field_name} must be non-empty")
        integer_fields = ("catalog_row_count", "embedding_dimension", "document_count")
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ManifestValidationError(f"manifest {field_name} must be an integer")
        if self.artifact_format_version != DENSE_ARTIFACT_FORMAT_VERSION:
            raise ManifestValidationError(
                f"unsupported dense artifact format: {self.artifact_format_version}"
            )
        if self.vector_dtype != DENSE_VECTOR_DTYPE:
            raise ManifestValidationError(f"unsupported vector dtype: {self.vector_dtype}")
        if self.normalization != DENSE_NORMALIZATION:
            raise ManifestValidationError(f"unsupported normalization: {self.normalization}")
        if self.catalog_row_count <= 0 or self.document_count <= 0:
            raise ManifestValidationError("manifest document counts must be positive")
        if self.catalog_row_count != self.document_count:
            raise ManifestValidationError("dense artifact must cover the complete catalog")
        if self.embedding_dimension <= 0:
            raise ManifestValidationError("embedding dimension must be positive")
        for field_name in ("catalog_source_sha256", "ordered_parent_asin_sha256"):
            checksum = getattr(self, field_name)
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ManifestValidationError(f"manifest {field_name} must be a lowercase SHA-256")
        if not self.index_id.startswith("dense-") or len(self.index_id) != 26:
            raise ManifestValidationError("manifest index_id has an invalid format")
        try:
            parsed_timestamp = datetime.fromisoformat(self.build_timestamp)
        except ValueError as error:
            raise ManifestValidationError("manifest build_timestamp must be ISO-8601") from error
        if parsed_timestamp.tzinfo is None:
            raise ManifestValidationError("manifest build_timestamp must include a timezone")
        if not isinstance(self.artifact_checksums, Mapping):
            raise ManifestValidationError("artifact_checksums must be a mapping")
        immutable_checksums = MappingProxyType(dict(self.artifact_checksums))
        object.__setattr__(self, "artifact_checksums", immutable_checksums)
        expected_artifacts = {DENSE_IDS_FILENAME, DENSE_VECTORS_FILENAME}
        if set(self.artifact_checksums) != expected_artifacts:
            raise ManifestValidationError(
                "artifact_checksums must contain exactly ids.jsonl and vectors.f32"
            )
        for filename, checksum in self.artifact_checksums.items():
            if not isinstance(filename, str) or not isinstance(checksum, str):
                raise ManifestValidationError("artifact checksum names and values must be strings")
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ManifestValidationError(f"invalid SHA-256 for {filename}")

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_format_version": self.artifact_format_version,
            "index_id": self.index_id,
            "catalog_source_sha256": self.catalog_source_sha256,
            "catalog_row_count": self.catalog_row_count,
            "ordered_parent_asin_sha256": self.ordered_parent_asin_sha256,
            "product_text_schema_version": self.product_text_schema_version,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_route_id": self.embedding_route_id,
            "embedding_dimension": self.embedding_dimension,
            "vector_dtype": self.vector_dtype,
            "normalization": self.normalization,
            "document_count": self.document_count,
            "build_timestamp": self.build_timestamp,
            "artifact_checksums": dict(sorted(self.artifact_checksums.items())),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "DenseIndexManifest":
        if not isinstance(payload, dict):
            raise ManifestValidationError("dense manifest must be a JSON object")
        required = {
            "artifact_format_version",
            "index_id",
            "catalog_source_sha256",
            "catalog_row_count",
            "ordered_parent_asin_sha256",
            "product_text_schema_version",
            "embedding_provider",
            "embedding_model",
            "embedding_route_id",
            "embedding_dimension",
            "vector_dtype",
            "normalization",
            "document_count",
            "build_timestamp",
            "artifact_checksums",
        }
        missing = required.difference(payload)
        unknown = set(payload).difference(required)
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
            return cls(
                artifact_format_version=payload["artifact_format_version"],
                index_id=payload["index_id"],
                catalog_source_sha256=payload["catalog_source_sha256"],
                catalog_row_count=payload["catalog_row_count"],
                ordered_parent_asin_sha256=payload["ordered_parent_asin_sha256"],
                product_text_schema_version=payload["product_text_schema_version"],
                embedding_provider=payload["embedding_provider"],
                embedding_model=payload["embedding_model"],
                embedding_route_id=payload["embedding_route_id"],
                embedding_dimension=payload["embedding_dimension"],
                vector_dtype=payload["vector_dtype"],
                normalization=payload["normalization"],
                document_count=payload["document_count"],
                build_timestamp=payload["build_timestamp"],
                artifact_checksums=MappingProxyType(dict(checksums)),
            )
        except (TypeError, ValueError, ManifestValidationError) as error:
            raise ManifestValidationError("dense manifest has invalid field types") from error

    def assert_compatible(
        self,
        catalog: ProductCatalog,
        *,
        embedding_route_id: str | None = None,
        index_id: str | None = None,
    ) -> None:
        mismatches: list[str] = []
        if self.catalog_source_sha256 != catalog.identity.source_sha256:
            mismatches.append("catalog_source_sha256")
        if self.catalog_row_count != catalog.identity.row_count:
            mismatches.append("catalog_row_count")
        if self.ordered_parent_asin_sha256 != catalog.identity.ordered_parent_asin_sha256:
            mismatches.append("ordered_parent_asin_sha256")
        if self.product_text_schema_version != PRODUCT_TEXT_SCHEMA_VERSION:
            mismatches.append("product_text_schema_version")
        expected_index_id = dense_index_id(
            catalog,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_route_id=self.embedding_route_id,
            embedding_dimension=self.embedding_dimension,
        )
        if self.index_id != expected_index_id:
            mismatches.append("derived_index_id")
        if embedding_route_id is not None and self.embedding_route_id != embedding_route_id:
            mismatches.append("embedding_route_id")
        if index_id is not None and self.index_id != index_id:
            mismatches.append("index_id")
        if mismatches:
            raise ManifestValidationError(
                "dense index is incompatible: " + ", ".join(mismatches)
            )


def dense_index_id(
    catalog: ProductCatalog,
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_route_id: str,
    embedding_dimension: int,
) -> str:
    identity = {
        "artifact_format_version": DENSE_ARTIFACT_FORMAT_VERSION,
        "catalog_source_sha256": catalog.identity.source_sha256,
        "catalog_row_count": catalog.identity.row_count,
        "ordered_parent_asin_sha256": catalog.identity.ordered_parent_asin_sha256,
        "product_text_schema_version": PRODUCT_TEXT_SCHEMA_VERSION,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_route_id": embedding_route_id,
        "embedding_dimension": embedding_dimension,
        "vector_dtype": DENSE_VECTOR_DTYPE,
        "normalization": DENSE_NORMALIZATION,
        "document_count": len(catalog),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "dense-" + hashlib.sha256(encoded).hexdigest()[:20]
