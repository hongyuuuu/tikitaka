"""One call for orchestration to obtain an interpreter.

Person 4 should not have to know whether a credential exists, which provider
is configured, or how to assemble a transport. It asks for an interpreter and
gets one, plus the route identity to record in the experiment report.

The credential-absent case is not an error. It is the network-free route that
M5 requires, so a missing key degrades the agent's quality rather than its
validity.
"""

from __future__ import annotations

import os
from typing import Mapping

from tikitaka.models.api_llm import ApiConfig, ApiInterpreter
from tikitaka.models.base import CredentialMissing, ModelRoute
from tikitaka.models.fake import HEURISTIC_ROUTE, HeuristicInterpreter
from tikitaka.models.http_transport import (
    CREDENTIAL_VARIABLE,
    HttpTransport,
    HttpTransportConfig,
)

PRIMARY_ROUTE = ModelRoute(
    route_id="primary/gpt-5.6-terra",
    provider="openai",
    model="gpt-5.6-terra",
    reasoning_level="xhigh",
)


def interpreter_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    route: ModelRoute = PRIMARY_ROUTE,
    api_config: ApiConfig | None = None,
    transport_config: HttpTransportConfig | None = None,
    credential_variable: str = CREDENTIAL_VARIABLE,
    allow_degraded: bool = True,
) -> tuple[object, str]:
    """Return `(interpreter, route_id)`.

    With `allow_degraded` false a missing credential raises instead, which is
    what a live evaluation run should use so it fails loudly rather than
    quietly scoring the deterministic route and reporting it as the API one.
    """

    environ = os.environ if environ is None else environ
    credential = (environ.get(credential_variable) or "").strip()

    if not credential:
        if not allow_degraded:
            raise CredentialMissing(
                f"environment variable {credential_variable} is unset", route
            )
        return HeuristicInterpreter(), HEURISTIC_ROUTE.route_id

    transport = HttpTransport(credential, route, transport_config)
    config = api_config or ApiConfig(route=route)
    return ApiInterpreter(transport, config, credential=credential), route.route_id


def describe_route(
    environ: Mapping[str, str] | None = None,
    *,
    credential_variable: str = CREDENTIAL_VARIABLE,
) -> dict[str, object]:
    """Reportable route facts. Never includes the credential itself."""

    environ = os.environ if environ is None else environ
    configured = bool((environ.get(credential_variable) or "").strip())
    route = PRIMARY_ROUTE if configured else HEURISTIC_ROUTE
    return {
        "route_id": route.route_id,
        "provider": route.provider,
        "model": route.model,
        "reasoning_level": route.reasoning_level,
        "credential_variable": credential_variable,
        "credential_present": configured,
        "degraded": not configured,
    }


__all__ = ["PRIMARY_ROUTE", "describe_route", "interpreter_from_env"]
