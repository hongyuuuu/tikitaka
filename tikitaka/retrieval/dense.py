"""Provider-neutral dense artifact building and exact cosine retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Sequence

from tikitaka.contracts import IndexManifest

from .catalog import ProductCatalog
from .manifests import (
    DENSE_ARTIFACT_FORMAT,
    DENSE_CHECKPOINT_FILENAME,
    DENSE_IDS_FILENAME,
    DENSE_MANIFEST_FILENAME,
    DENSE_NORMALIZED,
    DENSE_VECTORS_FILENAME,
    DENSE_VECTOR_DTYPE,
    ManifestValidationError,
    assert_dense_manifest_compatible,
    dense_manifest_as_dict,
    dense_manifest_from_dict,
    dense_index_id,
)
from .text import PRODUCT_TEXT_SCHEMA_VERSION, build_dense_text


class DenseArtifactError(RuntimeError):
    """Raised when a dense artifact cannot be safely built or loaded."""


class DenseRouteError(RuntimeError):
    """Raised when a query embedder cannot safely serve a dense index."""


def _declared_identity(embedder: object, name: str) -> str | None:
    value = getattr(embedder, name, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DenseRouteError(f"embedder {name} must be a non-empty string when declared")
    if value != value.strip():
        raise DenseRouteError(f"embedder {name} must not contain edge whitespace")
    return value


def assert_embedder_matches_manifest(embedder: object, manifest: IndexManifest) -> None:
    """Fail closed on every identity a provider-neutral embedder declares."""

    mismatches: list[str] = []
    declared = {
        "route_id": _route_id(embedder),
        "provider": _declared_identity(embedder, "provider"),
        "model": _declared_identity(embedder, "model"),
        "index_id": _declared_identity(embedder, "index_id"),
    }
    expected = {
        "route_id": manifest.route_id,
        "provider": manifest.provider,
        "model": manifest.model,
        "index_id": manifest.index_id,
    }
    for name, value in declared.items():
        if value is not None and value != expected[name]:
            mismatches.append(name)
    if mismatches:
        raise DenseRouteError(
            "query embedder does not match dense index manifest: " + ", ".join(mismatches)
        )


@dataclass(frozen=True, slots=True)
class DenseArtifactPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / DENSE_MANIFEST_FILENAME

    @property
    def ids(self) -> Path:
        return self.root / DENSE_IDS_FILENAME

    @property
    def vectors(self) -> Path:
        return self.root / DENSE_VECTORS_FILENAME

    @property
    def checkpoint(self) -> Path:
        return self.root / DENSE_CHECKPOINT_FILENAME

    @property
    def partial_ids(self) -> Path:
        return self.root / f"{DENSE_IDS_FILENAME}.partial"

    @property
    def partial_vectors(self) -> Path:
        return self.root / f"{DENSE_VECTORS_FILENAME}.partial"


@dataclass(frozen=True, slots=True)
class DenseHit:
    parent_asin: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class DenseLoadResult:
    index: DenseIndex | None
    failure_code: str | None


def _route_id(embedder: object) -> str:
    route_id = getattr(embedder, "route_id", "")
    if not isinstance(route_id, str) or not route_id.strip():
        raise DenseRouteError("embedder must expose a non-empty route_id")
    if route_id != route_id.strip():
        raise DenseRouteError("embedder route_id must not contain edge whitespace")
    return route_id


def _method(embedder: object, name: str) -> object:
    method = getattr(embedder, name, None)
    if not callable(method):
        raise DenseRouteError(f"embedder must implement {name}()")
    return method


def normalize_embedding(
    values: Iterable[object],
    *,
    expected_dimension: int | None = None,
) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise DenseRouteError("embedding contains a non-numeric value") from error
    if not vector:
        raise DenseRouteError("embedding vector must not be empty")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise DenseRouteError(
            f"embedding dimension mismatch: expected {expected_dimension}, found {len(vector)}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise DenseRouteError("embedding vector must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise DenseRouteError("embedding vector must have non-zero norm")
    return tuple(value / norm for value in vector)


def _float32_bytes(vector: Sequence[float]) -> bytes:
    values = array("f", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path, *, label: str) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise DenseArtifactError(f"{label} not found: {path}") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise DenseArtifactError(f"malformed {label}: {path}") from error


def _checkpoint_identity(
    catalog: ProductCatalog,
    *,
    provider: str,
    model: str,
    route_id: str,
    batch_size: int,
) -> dict[str, object]:
    return {
        "checkpoint_version": 1,
        "artifact_format_version": DENSE_ARTIFACT_FORMAT,
        "catalog_source_sha256": catalog.identity.source_sha256,
        "catalog_row_count": len(catalog),
        "ordered_parent_asin_sha256": catalog.identity.ordered_parent_asin_sha256,
        "product_text_schema_version": PRODUCT_TEXT_SCHEMA_VERSION,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_route_id": route_id,
        "batch_size": batch_size,
    }


def _validate_resume(
    paths: DenseArtifactPaths,
    catalog: ProductCatalog,
    expected_identity: dict[str, object],
) -> tuple[int, int | None, bool]:
    if not paths.checkpoint.exists():
        if any(
            path.exists()
            for path in (paths.partial_ids, paths.partial_vectors, paths.ids, paths.vectors)
        ):
            raise DenseArtifactError(
                "dense files exist without a manifest or checkpoint; use a new output directory"
            )
        return 0, None, False
    payload = _read_json(paths.checkpoint, label="dense build checkpoint")
    if not isinstance(payload, dict):
        raise DenseArtifactError("dense build checkpoint must be a JSON object")
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise DenseArtifactError(f"dense build checkpoint mismatch: {key}")
    try:
        next_index = int(payload["next_index"])
        dimension_raw = payload.get("embedding_dimension")
        dimension = None if dimension_raw is None else int(dimension_raw)
    except (KeyError, TypeError, ValueError) as error:
        raise DenseArtifactError("dense build checkpoint has invalid progress fields") from error
    if not 0 <= next_index <= len(catalog):
        raise DenseArtifactError("dense build checkpoint next_index is out of range")
    if next_index and (dimension is None or dimension <= 0):
        raise DenseArtifactError("dense build checkpoint is missing its embedding dimension")
    partial_pair = paths.partial_ids.is_file() and paths.partial_vectors.is_file()
    final_pair = paths.ids.is_file() and paths.vectors.is_file()
    recoverable_mid_rename = (
        paths.ids.is_file()
        and paths.partial_vectors.is_file()
        and not paths.partial_ids.exists()
        and not paths.vectors.exists()
    )
    if recoverable_mid_rename and next_index == len(catalog):
        paths.partial_vectors.replace(paths.vectors)
        final_pair = True
    if partial_pair == final_pair:
        raise DenseArtifactError(
            "dense checkpoint must have exactly one complete partial or final artifact pair"
        )
    if final_pair and next_index != len(catalog):
        raise DenseArtifactError("final dense files exist before the checkpoint reached completion")
    id_path = paths.ids if final_pair else paths.partial_ids
    vector_path = paths.vectors if final_pair else paths.partial_vectors
    with id_path.open(encoding="utf-8") as handle:
        try:
            identifiers = tuple(json.loads(line) for line in handle if line.strip())
        except json.JSONDecodeError as error:
            raise DenseArtifactError("partial dense ID artifact is malformed") from error
    expected_ids = tuple(product.parent_asin for product in catalog.products[:next_index])
    if identifiers != expected_ids:
        raise DenseArtifactError("partial dense IDs do not match the catalog prefix")
    expected_bytes = next_index * int(dimension or 0) * 4
    if vector_path.stat().st_size != expected_bytes:
        raise DenseArtifactError("partial dense vector size does not match the checkpoint")
    return next_index, dimension, final_pair


def build_dense_artifact(
    catalog: ProductCatalog,
    embedder: object,
    output_directory: str | Path,
    *,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int = 128,
) -> IndexManifest:
    """Build a complete resumable float32 artifact via the shared Embedder shape."""

    if batch_size <= 0:
        raise ValueError("embedding batch_size must be positive")
    provider = embedding_provider.strip()
    model = embedding_model.strip()
    if not provider or not model:
        raise ValueError("embedding provider and model must be non-empty")
    route_id = _route_id(embedder)
    declared_provider = _declared_identity(embedder, "provider")
    declared_model = _declared_identity(embedder, "model")
    declared_index_id = _declared_identity(embedder, "index_id")
    if declared_provider is not None and declared_provider != provider:
        raise DenseRouteError("embedder provider does not match requested artifact provider")
    if declared_model is not None and declared_model != model:
        raise DenseRouteError("embedder model does not match requested artifact model")
    embed_documents = _method(embedder, "embed_documents")
    paths = DenseArtifactPaths(Path(output_directory))
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.manifest.exists():
        manifest = read_dense_manifest(paths.root)
        if manifest.provider != provider or manifest.model != model:
            raise DenseArtifactError(
                "existing dense artifact provider/model does not match the requested build"
            )
        assert_embedder_matches_manifest(embedder, manifest)
        loaded = load_dense_index(
            paths.root,
            catalog,
            embedding_route_id=route_id,
            index_id=manifest.index_id,
        )
        if paths.partial_ids.exists() or paths.partial_vectors.exists():
            raise DenseArtifactError("completed dense artifact has unexpected partial files")
        paths.checkpoint.unlink(missing_ok=True)
        return loaded.manifest
    identity = _checkpoint_identity(
        catalog,
        provider=provider,
        model=model,
        route_id=route_id,
        batch_size=batch_size,
    )
    resuming = paths.checkpoint.exists()
    next_index, dimension, finalized = _validate_resume(paths, catalog, identity)
    if dimension is not None and declared_index_id is not None:
        expected_index_id = dense_index_id(
            catalog,
            embedding_provider=provider,
            embedding_model=model,
            embedding_route_id=route_id,
            embedding_dimension=dimension,
        )
        if declared_index_id != expected_index_id:
            raise DenseRouteError(
                "embedder index_id does not match the artifact being resumed"
            )
    if not resuming:
        paths.partial_ids.touch(exist_ok=False)
        paths.partial_vectors.touch(exist_ok=False)
        checkpoint = {**identity, "next_index": 0, "embedding_dimension": None}
        _write_json_atomic(paths.checkpoint, checkpoint)

    if not finalized:
        with paths.partial_ids.open("a", encoding="utf-8") as id_handle, paths.partial_vectors.open(
            "ab"
        ) as vector_handle:
            for start in range(next_index, len(catalog), batch_size):
                products = catalog.products[start : start + batch_size]
                texts = tuple(build_dense_text(product) for product in products)
                try:
                    raw_batch = tuple(embed_documents(texts))  # type: ignore[operator]
                except Exception as error:
                    raise DenseRouteError(
                        f"document embedding batch failed at catalog offset {start}"
                    ) from error
                if len(raw_batch) != len(products):
                    raise DenseRouteError(
                        f"embedder returned {len(raw_batch)} vectors for {len(products)} documents"
                    )
                normalized_batch: list[tuple[float, ...]] = []
                discovered_dimension = dimension is None
                for raw_vector in raw_batch:
                    normalized = normalize_embedding(raw_vector, expected_dimension=dimension)
                    if dimension is None:
                        dimension = len(normalized)
                    normalized_batch.append(normalized)
                if discovered_dimension and declared_index_id is not None:
                    expected_index_id = dense_index_id(
                        catalog,
                        embedding_provider=provider,
                        embedding_model=model,
                        embedding_route_id=route_id,
                        embedding_dimension=int(dimension),
                    )
                    if declared_index_id != expected_index_id:
                        raise DenseRouteError(
                            "embedder index_id does not match the artifact being built"
                        )
                for product, vector in zip(products, normalized_batch):
                    id_handle.write(json.dumps(product.parent_asin, ensure_ascii=True) + "\n")
                    vector_handle.write(_float32_bytes(vector))
                id_handle.flush()
                vector_handle.flush()
                os.fsync(id_handle.fileno())
                os.fsync(vector_handle.fileno())
                checkpoint = {
                    **identity,
                    "next_index": start + len(products),
                    "embedding_dimension": dimension,
                }
                _write_json_atomic(paths.checkpoint, checkpoint)

    if dimension is None:
        raise DenseArtifactError("cannot build a dense artifact for an empty catalog")
    if not finalized:
        if paths.ids.exists() or paths.vectors.exists():
            raise DenseArtifactError("final dense artifact files already exist without a manifest")
        paths.partial_ids.replace(paths.ids)
        paths.partial_vectors.replace(paths.vectors)
    checksums = MappingProxyType(
        {
            DENSE_IDS_FILENAME: _sha256(paths.ids),
            DENSE_VECTORS_FILENAME: _sha256(paths.vectors),
        }
    )
    manifest = IndexManifest(
        index_id=dense_index_id(
            catalog,
            embedding_provider=provider,
            embedding_model=model,
            embedding_route_id=route_id,
            embedding_dimension=dimension,
        ),
        catalog_checksum=catalog.identity.source_sha256,
        catalog_row_count=len(catalog),
        ordered_id_checksum=catalog.identity.ordered_parent_asin_sha256,
        product_text_schema_version=PRODUCT_TEXT_SCHEMA_VERSION,
        provider=provider,
        model=model,
        route_id=route_id,
        dimension=dimension,
        vector_dtype=DENSE_VECTOR_DTYPE,
        normalized=DENSE_NORMALIZED,
        document_count=len(catalog),
        artifact_format=DENSE_ARTIFACT_FORMAT,
        built_at=datetime.now(timezone.utc).isoformat(),
        artifact_checksums=checksums,
    )
    assert_embedder_matches_manifest(embedder, manifest)
    _write_json_atomic(paths.manifest, dense_manifest_as_dict(manifest))
    verified = load_dense_index(
        paths.root,
        catalog,
        embedding_route_id=route_id,
        index_id=manifest.index_id,
    )
    paths.checkpoint.unlink(missing_ok=True)
    return verified.manifest


def read_dense_manifest(directory: str | Path) -> IndexManifest:
    paths = DenseArtifactPaths(Path(directory))
    payload = _read_json(paths.manifest, label="dense manifest")
    try:
        return dense_manifest_from_dict(payload)
    except ManifestValidationError as error:
        raise DenseArtifactError(str(error)) from error


def _verify_artifact_checksums(
    paths: DenseArtifactPaths,
    manifest: IndexManifest,
) -> None:
    artifacts = {
        DENSE_IDS_FILENAME: paths.ids,
        DENSE_VECTORS_FILENAME: paths.vectors,
    }
    for filename, path in artifacts.items():
        if not path.is_file():
            raise DenseArtifactError(f"dense artifact file not found: {path}")
        if _sha256(path) != manifest.artifact_checksums[filename]:
            raise DenseArtifactError(f"dense artifact checksum mismatch: {filename}")


class DenseIndex:
    """Immutable full-catalog normalized vector matrix with exact cosine search."""

    def __init__(
        self,
        catalog: ProductCatalog,
        manifest: IndexManifest,
        identifiers: tuple[str, ...],
        vectors: object,
        *,
        backend: str,
    ) -> None:
        self.catalog = catalog
        self.manifest = manifest
        self.identifiers = identifiers
        self._vectors = vectors
        self.backend = backend
        self._numpy_ids: object | None = None

    def search(self, query_vector: Iterable[object], *, limit: int = 200) -> list[DenseHit]:
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        query = normalize_embedding(
            query_vector,
            expected_dimension=self.manifest.dimension,
        )
        if self.backend == "numpy-exact":
            try:
                import numpy as np
            except ImportError as error:
                raise DenseArtifactError(
                    "NumPy dense index was loaded but NumPy is no longer available"
                ) from error
            matrix = self._vectors
            query_array = np.asarray(query, dtype=np.float32)
            scores = matrix @ query_array  # type: ignore[operator]
            if self._numpy_ids is None:
                self._numpy_ids = np.asarray(self.identifiers)
            order = np.lexsort((self._numpy_ids, -scores))[:limit]
            return [
                DenseHit(
                    parent_asin=self.identifiers[int(index)],
                    rank=rank,
                    score=float(scores[int(index)]),
                )
                for rank, index in enumerate(order, start=1)
            ]
        dimension = self.manifest.dimension
        scores: list[tuple[str, float]] = []
        for index, parent_asin in enumerate(self.identifiers):
            offset = index * dimension
            score = sum(
                query[position] * self._vectors[offset + position]  # type: ignore[index]
                for position in range(dimension)
            )
            scores.append((parent_asin, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            DenseHit(parent_asin=parent_asin, rank=rank, score=score)
            for rank, (parent_asin, score) in enumerate(scores[:limit], start=1)
        ]


def load_dense_index(
    directory: str | Path,
    catalog: ProductCatalog,
    *,
    embedding_route_id: str | None = None,
    index_id: str | None = None,
    verify_normalization: bool = True,
) -> DenseIndex:
    paths = DenseArtifactPaths(Path(directory))
    manifest = read_dense_manifest(paths.root)
    try:
        assert_dense_manifest_compatible(
            manifest,
            catalog,
            embedding_route_id=embedding_route_id,
            index_id=index_id,
        )
    except ManifestValidationError as error:
        raise DenseArtifactError(str(error)) from error
    _verify_artifact_checksums(paths, manifest)
    identifiers: list[str] = []
    try:
        with paths.ids.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DenseArtifactError(
                        f"malformed dense ID at line {line_number}"
                    ) from error
                if not isinstance(value, str) or not value:
                    raise DenseArtifactError(f"invalid dense ID at line {line_number}")
                identifiers.append(value)
    except UnicodeError as error:
        raise DenseArtifactError("dense ID artifact is not valid UTF-8") from error
    expected_ids = tuple(product.parent_asin for product in catalog)
    if tuple(identifiers) != expected_ids:
        raise DenseArtifactError("dense ID order does not match the frozen catalog")
    expected_floats = manifest.document_count * manifest.dimension
    expected_bytes = expected_floats * 4
    if paths.vectors.stat().st_size != expected_bytes:
        raise DenseArtifactError("dense vector file size does not match its manifest")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        vectors = np.memmap(
            paths.vectors,
            dtype="<f4",
            mode="r",
            shape=(manifest.document_count, manifest.dimension),
        )
        if verify_normalization:
            if not bool(np.isfinite(vectors).all()):
                raise DenseArtifactError("dense vectors contain non-finite values")
            norm_squared = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
            if not bool(np.allclose(norm_squared, 1.0, rtol=2e-5, atol=2e-5)):
                bad_index = int(np.flatnonzero(~np.isclose(norm_squared, 1.0, rtol=2e-5, atol=2e-5))[0])
                raise DenseArtifactError(f"dense vector {bad_index} is not L2 normalized")
        return DenseIndex(
            catalog,
            manifest,
            tuple(identifiers),
            vectors,
            backend="numpy-exact",
        )
    vectors_array = array("f")
    with paths.vectors.open("rb") as handle:
        vectors_array.fromfile(handle, expected_floats)
    if sys.byteorder != "little":
        vectors_array.byteswap()
    if len(vectors_array) != expected_floats:
        raise DenseArtifactError("dense vector file ended before the declared document count")
    if verify_normalization:
        dimension = manifest.dimension
        for document_index in range(manifest.document_count):
            offset = document_index * dimension
            norm_squared = sum(
                vectors_array[offset + position] * vectors_array[offset + position]
                for position in range(dimension)
            )
            if not math.isclose(norm_squared, 1.0, rel_tol=2e-5, abs_tol=2e-5):
                raise DenseArtifactError(
                    f"dense vector {document_index} is not L2 normalized"
                )
    return DenseIndex(
        catalog,
        manifest,
        tuple(identifiers),
        vectors_array,
        backend="python-exact",
    )


def load_dense_index_safe(
    directory: str | Path,
    catalog: ProductCatalog,
    *,
    embedding_route_id: str | None = None,
    index_id: str | None = None,
) -> DenseLoadResult:
    """Load an optional dense artifact without making agent startup fatal."""

    root = Path(directory)
    if not (root / DENSE_MANIFEST_FILENAME).is_file():
        return DenseLoadResult(index=None, failure_code="dense_artifact_unavailable")
    try:
        index = load_dense_index(
            root,
            catalog,
            embedding_route_id=embedding_route_id,
            index_id=index_id,
        )
    except (DenseArtifactError, OSError):
        return DenseLoadResult(index=None, failure_code="dense_artifact_invalid")
    return DenseLoadResult(index=index, failure_code=None)


def embed_query_for_index(embedder: object, index: DenseIndex, text: str) -> tuple[float, ...]:
    assert_embedder_matches_manifest(embedder, index.manifest)
    embed_query = _method(embedder, "embed_query")
    try:
        raw_vector = embed_query(text)  # type: ignore[operator]
    except Exception as error:
        raise DenseRouteError("query embedding call failed") from error
    return normalize_embedding(
        raw_vector,
        expected_dimension=index.manifest.dimension,
    )
