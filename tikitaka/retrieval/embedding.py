"""Bridge Person 1's batch model gateway to the shared retrieval Embedder.

The adapter contains no provider SDK or credential handling. It binds one
provider-neutral ``EmbeddingModel`` to one immutable ``ModelRoute``, validates
usage attribution, and exposes the document/query methods required by Role 2.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Sequence

from tikitaka.contracts import EmbeddingBatch, EmbeddingVector, IndexManifest, Usage
from tikitaka.models.base import EmbeddingModel, ModelRoute
from tikitaka.models.usage import merge


class EmbeddingAdapterError(ValueError):
    """Raised when a gateway route or response violates the embedding contract."""


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingAdapterError(f"embedding route {name} must be a non-empty string")
    if value != value.strip():
        raise EmbeddingAdapterError(f"embedding route {name} must not contain edge whitespace")
    return value


def _validate_route(route: ModelRoute) -> None:
    if not isinstance(route, ModelRoute):
        raise EmbeddingAdapterError("route must be a ModelRoute")
    _non_empty(route.route_id, "route_id")
    _non_empty(route.provider, "provider")
    _non_empty(route.model, "model")
    if route.index_id is not None:
        _non_empty(route.index_id, "index_id")


def _validate_usage(usage: object, route: ModelRoute) -> Usage:
    if not isinstance(usage, Usage):
        raise EmbeddingAdapterError("embedding model must return canonical Usage")
    if usage.calls == 0 and not usage.cache_hit:
        raise EmbeddingAdapterError(
            "embedding usage must report a call or an attributable cache hit"
        )
    if usage.calls > 0 or usage.cache_hit:
        expected = {
            "provider": route.provider,
            "model": route.model,
            "route": route.route_id,
            "reasoning_level": route.reasoning_level,
        }
        mismatches = [
            name for name, value in expected.items() if getattr(usage, name) != value
        ]
        if mismatches:
            raise EmbeddingAdapterError(
                "embedding usage identity mismatch: " + ", ".join(mismatches)
            )
    return usage


def _vectors(payload: object, expected_count: int) -> EmbeddingBatch:
    if isinstance(payload, (str, bytes)):
        raise EmbeddingAdapterError("embedding model returned a malformed vector batch")
    try:
        raw_vectors = tuple(payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise EmbeddingAdapterError("embedding model returned a malformed vector batch") from error
    if len(raw_vectors) != expected_count:
        raise EmbeddingAdapterError(
            f"embedding model returned {len(raw_vectors)} vectors for {expected_count} texts"
        )
    vectors: list[EmbeddingVector] = []
    dimension: int | None = None
    for raw_vector in raw_vectors:
        if isinstance(raw_vector, (str, bytes)):
            raise EmbeddingAdapterError("embedding model returned a malformed vector")
        try:
            vector = tuple(float(value) for value in raw_vector)  # type: ignore[union-attr]
        except (TypeError, ValueError) as error:
            raise EmbeddingAdapterError("embedding model returned a malformed vector") from error
        if not vector:
            raise EmbeddingAdapterError("embedding model returned an empty vector")
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingAdapterError("embedding model returned a non-finite vector")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingAdapterError("embedding model returned inconsistent dimensions")
        vectors.append(vector)
    return tuple(vectors)


def embedding_usage_as_dict(usage: Usage) -> dict[str, object]:
    """Serialize attributable embedding usage without provider-specific data."""

    if not isinstance(usage, Usage):
        raise TypeError("usage must be canonical Usage")
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "calls": usage.calls,
        "repairs": usage.repairs,
        "latency_ms": usage.latency_ms,
        "provider": usage.provider,
        "model": usage.model,
        "reasoning_level": usage.reasoning_level,
        "estimated_cost": usage.estimated_cost,
        "cost_currency": usage.cost_currency,
        "route": usage.route,
        "cache_hit": usage.cache_hit,
    }


class GatewayEmbedder:
    """Adapt a provider-neutral batch model to the canonical Embedder protocol."""

    def __init__(self, model: EmbeddingModel, route: ModelRoute) -> None:
        _validate_route(route)
        embed = getattr(model, "embed", None)
        if not callable(embed):
            raise EmbeddingAdapterError("embedding model must implement embed(texts, route)")
        self._model = model
        self._route = route
        self._usage: Usage | None = None
        self._usage_lock = Lock()

    @property
    def route_id(self) -> str:
        return self._route.route_id

    @property
    def provider(self) -> str:
        return self._route.provider

    @property
    def model(self) -> str:
        return self._route.model

    @property
    def index_id(self) -> str | None:
        return self._route.index_id

    @property
    def usage(self) -> Usage:
        with self._usage_lock:
            return self._usage or Usage()

    def take_usage(self) -> Usage:
        """Atomically return and clear usage accumulated since the last take."""

        with self._usage_lock:
            usage = self._usage or Usage()
            self._usage = None
            return usage

    def assert_manifest(self, manifest: IndexManifest) -> None:
        mismatches: list[str] = []
        if manifest.route_id != self.route_id:
            mismatches.append("route_id")
        if manifest.provider != self.provider:
            mismatches.append("provider")
        if manifest.model != self.model:
            mismatches.append("model")
        if self.index_id is not None and manifest.index_id != self.index_id:
            mismatches.append("index_id")
        if mismatches:
            raise EmbeddingAdapterError(
                "embedding route does not match index manifest: " + ", ".join(mismatches)
            )

    def _embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        normalized = tuple(texts)
        if not all(isinstance(text, str) for text in normalized):
            raise EmbeddingAdapterError("embedding inputs must contain only strings")
        if not normalized:
            return ()
        payload, usage = self._model.embed(normalized, self._route)
        checked_usage = _validate_usage(usage, self._route)
        with self._usage_lock:
            if (
                self._usage is not None
                and self._usage.cost_currency != checked_usage.cost_currency
            ):
                raise EmbeddingAdapterError(
                    "embedding usage currency changed within one adapter"
                )
            self._usage = checked_usage if self._usage is None else merge(self._usage, checked_usage)
        return _vectors(payload, len(normalized))

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._embed(texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return self._embed((text,))[0]


__all__ = [
    "EmbeddingAdapterError",
    "GatewayEmbedder",
    "embedding_usage_as_dict",
]
