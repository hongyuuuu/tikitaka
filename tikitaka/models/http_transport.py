"""The provider-aware seam: HTTP, auth, and request shape.

This is the only module that knows the endpoint exists. Everything above it in
`api_llm.py` speaks `TransportResponse` and the provider-neutral error
taxonomy, which is what let the adapter be finished and tested before this
file existed.

Standard library only, deliberately. The repository has no dependency manifest
yet, so pulling in an SDK would make this module the project's first
third-party dependency — a coordination surface change for a few hundred lines
of JSON over HTTPS.

Request shape verified against a live `gpt-5.6-terra` response:

- text arrives at `choices[0].message.content`;
- `reasoning_effort: "xhigh"` is accepted and produces non-zero
  `completion_tokens_details.reasoning_tokens`;
- those reasoning tokens are a subset of `completion_tokens`, not an addition.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping

from tikitaka.models.api_llm import TransportResponse
from tikitaka.models.base import (
    MalformedModelOutput,
    ModelRefused,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
CREDENTIAL_VARIABLE = "OPENAI_API_KEY"

# Retryable on the provider side: rate limiting and transient server faults.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpTransportConfig:
    base_url: str = DEFAULT_BASE_URL
    reasoning_effort: str | None = "xhigh"
    # Structured output via a schema parameter is off until confirmed
    # supported for this model. The adapter re-validates locally either way,
    # so an ignored schema degrades safely rather than corrupting state.
    use_response_format: bool = False


class HttpTransport:
    """Sends one prompt to the configured chat-completions endpoint."""

    def __init__(
        self,
        credential: str,
        route: ModelRoute,
        config: HttpTransportConfig | None = None,
        *,
        opener=urllib.request.urlopen,
    ) -> None:
        if not (credential or "").strip():
            raise ValueError("credential must be a non-empty string")
        self._credential = credential
        self._route = route
        self._config = config or HttpTransportConfig()
        self._opener = opener

    def __repr__(self) -> str:
        return f"HttpTransport(route={self._route.route_id!r})"

    def send(
        self,
        prompt: str,
        schema: Mapping[str, object],
        timeout_s: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            self._config.base_url,
            data=json.dumps(self._body(prompt, schema)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self._opener(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise self._from_status(error) from None
        except socket.timeout:
            raise ModelTimeout("provider did not answer in time", self._route) from None
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, socket.timeout):
                raise ModelTimeout(
                    "provider did not answer in time", self._route
                ) from None
            raise ModelUnavailable(f"transport failure: {reason}", self._route) from None
        except (ValueError, TypeError) as error:
            raise MalformedModelOutput(
                f"provider response was not JSON: {error}", self._route
            ) from None

        return self._to_response(payload)

    # ---- internals ------------------------------------------------------

    def _body(self, prompt: str, schema: Mapping[str, object]) -> dict:
        body: dict = {
            "model": self._route.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        effort = self._config.reasoning_effort or self._route.reasoning_level
        if effort:
            body["reasoning_effort"] = effort
        if self._config.use_response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "state_delta",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        return body

    def _from_status(self, error: urllib.error.HTTPError) -> Exception:
        detail = ""
        try:
            detail = error.read().decode("utf-8")[:300]
        except Exception:  # pragma: no cover - diagnostic only
            detail = ""
        # Never echo the request, only the provider's own message: the auth
        # header must not reach a log through an error path.
        message = f"HTTP {error.code}: {detail}"
        if error.code in _TRANSIENT_STATUS:
            return ModelUnavailable(message, self._route)
        if error.code in (401, 403):
            return ModelRefused(f"authentication rejected (HTTP {error.code})", self._route)
        return ModelRefused(message, self._route)

    def _to_response(self, payload: Mapping[str, object]) -> TransportResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MalformedModelOutput("provider returned no choices", self._route)
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(message, dict) and message.get("refusal"):
            raise ModelRefused(str(message["refusal"])[:200], self._route)
        if not isinstance(content, str):
            raise MalformedModelOutput("provider returned no text", self._route)

        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, Mapping) else {}

        return TransportResponse(
            text=content,
            prompt_tokens=_count(usage.get("prompt_tokens")),
            completion_tokens=_count(usage.get("completion_tokens")),
            reasoning_tokens=_count(details.get("reasoning_tokens")),
        )


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = [
    "CREDENTIAL_VARIABLE",
    "DEFAULT_BASE_URL",
    "HttpTransport",
    "HttpTransportConfig",
]
