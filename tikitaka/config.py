"""Dependency-light routing configuration shared by runtime and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from tikitaka.contracts.domain import IndexManifest, RoutePolicy


CONTRACT_VERSION = "0.1.0"
STRUCTURED_OUTPUT_SCHEMA_VERSION = "0.1.0"


def _paired_embedding_identity(
    embedding_route_id: str | None,
    index_id: str | None,
) -> None:
    if (embedding_route_id is None) != (index_id is None):
        raise ValueError("embedding_route_id and index_id must be set together")


@dataclass(frozen=True)
class RuntimeRoutingConfig:
    """Automatic runtime routing defaults, independent of experiment pins."""

    generative_provider: str = "openai"
    generative_model: str = "gpt-5.6-terra"
    reasoning_level: str = "medium"
    retrieval_policy: RoutePolicy = RoutePolicy.AUTO
    embedding_route_id: str | None = None
    index_id: str | None = None
    reranker_route_id: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_policy", RoutePolicy(self.retrieval_policy))
        _paired_embedding_identity(self.embedding_route_id, self.index_id)

    def validate_index(self, manifest: IndexManifest) -> None:
        """Fail closed when a configured embedding route and index do not match."""

        if self.embedding_route_id is None:
            return
        if (
            manifest.route_id != self.embedding_route_id
            or manifest.index_id != self.index_id
        ):
            raise ValueError("configured embedding route does not match index manifest")


@dataclass(frozen=True)
class ExperimentPins:
    """Optional evaluation-only route pins; never mutated by runtime routing."""

    generative_route_id: str | None = None
    retrieval_policy: RoutePolicy | None = None
    embedding_route_id: str | None = None
    index_id: str | None = None
    reranker_route_id: str | None = None

    def __post_init__(self) -> None:
        if self.retrieval_policy is not None:
            object.__setattr__(self, "retrieval_policy", RoutePolicy(self.retrieval_policy))
        _paired_embedding_identity(self.embedding_route_id, self.index_id)
