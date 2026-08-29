"""Provider-neutral model surface: routes, protocols, and the error taxonomy.

Every error carries the route identity so failures stay attributable in the
experiment report. No error, `repr`, or log line may ever carry a credential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from tikitaka.contracts.domain import Usage


@dataclass(frozen=True)
class ModelRoute:
    """Identity of one configured model route.

    `index_id` is populated only for embedding routes and couples a query
    embedding to the product index built by the same model. The selector
    refuses a mismatch; it never compares vectors across embedding models.
    """

    route_id: str
    provider: str
    model: str
    reasoning_level: str | None = None
    index_id: str | None = None
    pinned: bool = False

    def __str__(self) -> str:
        return self.route_id


class ModelError(RuntimeError):
    """Base provider-neutral model failure."""

    def __init__(self, message: str, route: ModelRoute | None = None) -> None:
        super().__init__(message)
        self.route = route

    def __str__(self) -> str:
        base = super().__str__()
        if self.route is None:
            return base
        return f"{base} [route={self.route.route_id}]"


class ModelUnavailable(ModelError):
    """Transport failure, rate limit, or unreachable provider."""


class ModelTimeout(ModelError):
    """The provider did not answer within the configured budget."""


class ModelRefused(ModelError):
    """The provider declined to answer."""


class MalformedModelOutput(ModelError):
    """Output was not parseable as the requested structured payload."""


class SchemaViolation(ModelError):
    """Output parsed but violated the frozen structured-output schema."""


class CredentialMissing(ModelError):
    """No credential in the environment. Raised at construction, never mid-turn."""


class TextModel(Protocol):
    def complete_structured(
        self,
        prompt: str,
        schema: dict,
        route: ModelRoute,
    ) -> tuple[dict, Usage]: ...


class EmbeddingModel(Protocol):
    """Declared here for Person 2 to satisfy. Person 1 does not build indexes."""

    def embed(
        self,
        texts: Sequence[str],
        route: ModelRoute,
    ) -> tuple[list[list[float]], Usage]: ...


__all__ = [
    "CredentialMissing",
    "EmbeddingModel",
    "MalformedModelOutput",
    "ModelError",
    "ModelRefused",
    "ModelRoute",
    "ModelTimeout",
    "ModelUnavailable",
    "SchemaViolation",
    "TextModel",
]
