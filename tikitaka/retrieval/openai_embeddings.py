"""Production OpenAI embeddings route for dense and hybrid retrieval.

The implementation deliberately uses only the standard library, matching the
repository's existing HTTP transport. Credentials are accepted at construction
and never appear in reprs, exceptions, manifests, or experiment reports.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

from tikitaka.contracts import Usage
from tikitaka.models.base import (
    CredentialMissing,
    MalformedModelOutput,
    ModelRefused,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
)
from tikitaka.models.usage import for_route

from .embedding import GatewayEmbedder


DEFAULT_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

#: Person 2's production decision, 2026-08-30: accuracy first, at a width that
#: keeps the float32 index near 195 MB and therefore shippable inside a
#: submission bundle. The model's native 3072 would be 586 MB.
#:
#: This is a default rather than a caller's responsibility because the failure
#: is silent: leaving it unset sends no `dimensions` parameter at all, the
#: provider returns its native width, and the build produces a valid index of
#: the wrong size that every downstream identity check accepts. Override with
#: TIKITAKA_EMBEDDING_DIMENSIONS for the 512 packaging ablation.
PRODUCTION_EMBEDDING_DIMENSIONS = 1024
EMBEDDING_CREDENTIAL_VARIABLE = "OPENAI_API_KEY"
MAX_INPUTS_PER_REQUEST = 2_048
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return selected


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return selected


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain edge whitespace")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingConfig:
    """Immutable request and attribution settings for one embedding route."""

    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = PRODUCTION_EMBEDDING_DIMENSIONS
    base_url: str = DEFAULT_EMBEDDING_URL
    timeout_s: float = 60.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    input_cost_per_1m: float = 0.13

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(
            self,
            "base_url",
            _required_text(self.base_url, "base_url"),
        )
        object.__setattr__(
            self,
            "dimensions",
            _optional_positive_int(self.dimensions, "dimensions"),
        )
        object.__setattr__(
            self,
            "timeout_s",
            _positive_float(self.timeout_s, "timeout_s"),
        )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")
        object.__setattr__(
            self,
            "backoff_base_s",
            _non_negative_float(self.backoff_base_s, "backoff_base_s"),
        )
        object.__setattr__(
            self,
            "input_cost_per_1m",
            _non_negative_float(self.input_cost_per_1m, "input_cost_per_1m"),
        )

    @property
    def route_id(self) -> str:
        dimension = "default" if self.dimensions is None else str(self.dimensions)
        return f"openai/{self.model}/dimensions-{dimension}"


def openai_embedding_route(config: OpenAIEmbeddingConfig) -> ModelRoute:
    """Return the complete identity persisted in the dense manifest."""

    return ModelRoute(
        route_id=config.route_id,
        provider="openai",
        model=config.model,
        pinned=True,
    )


class OpenAIEmbeddingModel:
    """Provider implementation of Person 1's batch ``EmbeddingModel`` seam."""

    def __init__(
        self,
        credential: str,
        config: OpenAIEmbeddingConfig | None = None,
        *,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
    ) -> None:
        if not isinstance(credential, str) or not credential.strip():
            raise CredentialMissing("OpenAI embedding credential is missing")
        self._credential = credential
        self._config = config or OpenAIEmbeddingConfig()
        self._route = openai_embedding_route(self._config)
        self._opener = opener
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OpenAIEmbeddingModel(route={self._route.route_id!r})"

    def embed(
        self,
        texts: Sequence[str],
        route: ModelRoute,
    ) -> tuple[list[list[float]], Usage]:
        self._assert_route(route)
        if isinstance(texts, (str, bytes)):
            raise ValueError("embedding inputs must be a sequence of strings")
        normalized = tuple(texts)
        if not normalized:
            raise ValueError("embedding request must contain at least one input")
        if len(normalized) > MAX_INPUTS_PER_REQUEST:
            raise ValueError(
                f"embedding request exceeds {MAX_INPUTS_PER_REQUEST} inputs"
            )
        if not all(isinstance(text, str) and text.strip() for text in normalized):
            raise ValueError("embedding inputs must be non-empty strings")

        started = perf_counter()
        payload: Mapping[str, object] | None = None
        for attempt in range(self._config.max_attempts):
            try:
                payload = self._send(normalized)
                break
            except ModelUnavailable:
                if attempt + 1 >= self._config.max_attempts:
                    raise
                self._sleep(self._config.backoff_base_s * (2**attempt))
        if payload is None:  # pragma: no cover - loop invariant
            raise ModelUnavailable("embedding provider unavailable", route)

        vectors, prompt_tokens = self._parse(payload, len(normalized))
        usage = for_route(
            route,
            prompt_tokens=prompt_tokens,
            latency_ms=(perf_counter() - started) * 1_000.0,
            prompt_cost_per_1k=self._config.input_cost_per_1m / 1_000.0,
        )
        return vectors, usage

    def _assert_route(self, route: ModelRoute) -> None:
        expected = self._route
        if not isinstance(route, ModelRoute):
            raise ValueError("embedding route must be a ModelRoute")
        mismatches = [
            name
            for name in ("route_id", "provider", "model")
            if getattr(route, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ValueError(
                "OpenAI embedding route identity mismatch: "
                + ", ".join(mismatches)
            )

    def _send(self, texts: tuple[str, ...]) -> Mapping[str, object]:
        body: dict[str, object] = {
            "input": list(texts),
            "model": self._config.model,
            "encoding_format": "float",
        }
        if self._config.dimensions is not None:
            body["dimensions"] = self._config.dimensions
        request = urllib.request.Request(
            self._config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            translated = self._from_status(error)
            error.close()
            raise translated from None
        except socket.timeout:
            raise ModelTimeout(
                "embedding provider did not answer in time",
                self._route,
            ) from None
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, socket.timeout):
                raise ModelTimeout(
                    "embedding provider did not answer in time",
                    self._route,
                ) from None
            raise ModelUnavailable(
                f"embedding transport failure: {reason}",
                self._route,
            ) from None
        except (UnicodeError, ValueError, TypeError) as error:
            raise MalformedModelOutput(
                f"embedding provider response was not JSON: {error}",
                self._route,
            ) from None
        if not isinstance(payload, Mapping):
            raise MalformedModelOutput(
                "embedding provider returned a non-object response",
                self._route,
            )
        return payload

    def _from_status(self, error: urllib.error.HTTPError) -> Exception:
        if error.code in _TRANSIENT_STATUS:
            return ModelUnavailable(
                f"embedding provider unavailable (HTTP {error.code})",
                self._route,
            )
        if error.code in (401, 403):
            return ModelRefused(
                f"embedding authentication rejected (HTTP {error.code})",
                self._route,
            )
        return ModelRefused(
            f"embedding request rejected (HTTP {error.code})",
            self._route,
        )

    def _parse(
        self,
        payload: Mapping[str, object],
        expected_count: int,
    ) -> tuple[list[list[float]], int]:
        response_model = payload.get("model")
        if response_model != self._config.model:
            raise MalformedModelOutput(
                "embedding provider returned an unexpected model identity",
                self._route,
            )
        raw_data = payload.get("data")
        if not isinstance(raw_data, list):
            raise MalformedModelOutput(
                "embedding provider returned no vector data",
                self._route,
            )
        if len(raw_data) != expected_count:
            raise MalformedModelOutput(
                "embedding provider returned the wrong number of vectors",
                self._route,
            )
        indexed: dict[int, list[float]] = {}
        for item in raw_data:
            if not isinstance(item, Mapping):
                raise MalformedModelOutput(
                    "embedding provider returned malformed vector data",
                    self._route,
                )
            index = item.get("index")
            vector = item.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(raw_data)
                or index in indexed
                or not isinstance(vector, list)
            ):
                raise MalformedModelOutput(
                    "embedding provider returned invalid vector indexes",
                    self._route,
                )
            indexed[index] = vector
        if set(indexed) != set(range(len(raw_data))):
            raise MalformedModelOutput(
                "embedding provider returned incomplete vector indexes",
                self._route,
            )
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            raise MalformedModelOutput(
                "embedding provider returned no usage",
                self._route,
            )
        prompt_tokens = usage.get("prompt_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise MalformedModelOutput(
                "embedding provider returned invalid usage",
                self._route,
            )
        vectors = [indexed[index] for index in range(len(indexed))]
        self._assert_dimensions(vectors)
        return vectors, prompt_tokens

    def _assert_dimensions(self, vectors: Sequence[Sequence[float]]) -> None:
        """Refuse vectors that are not the width the request asked for.

        `dimensions` is a request parameter, not a guarantee: a provider may
        ignore it and return the model's native width. Nothing downstream
        notices. `build_dense_index` learns the width from the first batch, so
        it would record whatever arrived, and the manifest, the checksums and
        every identity assertion would then agree with each other — a
        internally consistent index of the wrong shape, three times the agreed
        size, discoverable only by looking at the file.
        """

        expected = self._config.dimensions
        if expected is None:
            return
        for vector in vectors:
            if len(vector) != expected:
                raise MalformedModelOutput(
                    f"embedding provider returned {len(vector)}-dimensional "
                    f"vectors for a request of {expected}; the dimensions "
                    "parameter was ignored or is unsupported for this model",
                    self._route,
                )


def _environment_int(
    environ: Mapping[str, str],
    name: str,
    default: int | None,
) -> int | None:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    return value


def _environment_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error


def openai_embedder_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    opener=urllib.request.urlopen,
    sleep=time.sleep,
) -> GatewayEmbedder:
    """Build the production embedder used by index and sweep CLIs.

    Supported overrides are intentionally explicit so every value that changes
    vector identity or API behavior can be reproduced in a report.
    """

    selected = os.environ if environ is None else environ
    defaults = OpenAIEmbeddingConfig()
    config = OpenAIEmbeddingConfig(
        model=selected.get("TIKITAKA_EMBEDDING_MODEL", defaults.model),
        dimensions=_environment_int(
            selected,
            "TIKITAKA_EMBEDDING_DIMENSIONS",
            defaults.dimensions,
        ),
        base_url=selected.get("TIKITAKA_EMBEDDING_BASE_URL", defaults.base_url),
        timeout_s=_environment_float(
            selected,
            "TIKITAKA_EMBEDDING_TIMEOUT_S",
            defaults.timeout_s,
        ),
        max_attempts=_environment_int(
            selected,
            "TIKITAKA_EMBEDDING_MAX_ATTEMPTS",
            defaults.max_attempts,
        ),
        backoff_base_s=_environment_float(
            selected,
            "TIKITAKA_EMBEDDING_BACKOFF_BASE_S",
            defaults.backoff_base_s,
        ),
        input_cost_per_1m=_environment_float(
            selected,
            "TIKITAKA_EMBEDDING_INPUT_COST_PER_1M",
            defaults.input_cost_per_1m,
        ),
    )
    credential = selected.get(EMBEDDING_CREDENTIAL_VARIABLE, "")
    model = OpenAIEmbeddingModel(
        credential,
        config,
        opener=opener,
        sleep=sleep,
    )
    return GatewayEmbedder(model, openai_embedding_route(config))


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_URL",
    "EMBEDDING_CREDENTIAL_VARIABLE",
    "MAX_INPUTS_PER_REQUEST",
    "PRODUCTION_EMBEDDING_DIMENSIONS",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingModel",
    "openai_embedder_from_env",
    "openai_embedding_route",
]
