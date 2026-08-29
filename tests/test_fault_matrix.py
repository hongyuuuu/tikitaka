"""P5: the fault matrix, run through the real orchestrator.

Every fault a model route can produce is driven through `build_agent` and the
official `respond` contract. The bar is not that the agent answers well under
failure — it is that it answers *validly*, because the evaluator scores a
malformed response as a miss and an exception as a zero.

Nothing here touches the network, a credential, or the frozen catalog.
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import socket
import unittest
import urllib.error

from tikitaka.contracts.domain import Usage
from tikitaka.models.api_llm import ApiConfig, ApiInterpreter, TransportResponse
from tikitaka.models.base import (
    CredentialMissing,
    ModelRefused,
    ModelTimeout,
    ModelUnavailable,
)
from tikitaka.models.factory import PRIMARY_ROUTE, GatewaySelection, gateway_from_env
from tikitaka.models.http_transport import HttpTransport
from tikitaka.orchestration.runtime import RuntimeConfig, build_agent
from tikitaka.state.schema import MAX_OPERATIONS, parse
from tikitaka.state.session import new_session
from tikitaka.state.trace import REDACTED, capture, redact

CATALOG = pathlib.Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"
CATALOG_IDS = frozenset({"TINY-A", "TINY-B", "TINY-C"})
SECRET = "sk-not-a-real-key-0123456789"

ALLOWED_ATTRIBUTES = frozenset(
    {
        "category", "material", "color", "size", "style", "brand",
        "budget", "feature", "use_case", "other",
    }
)

WELL_FORMED = json.dumps(
    {
        "inferred_mode": "buying",
        "mode_confidence": 0.9,
        "generality": 0.2,
        "operations": [
            {
                "operation": "add",
                "attribute": "material",
                "new_value": "cotton",
                "polarity": "include",
                "strength": "soft",
                "confidence": 0.8,
            }
        ],
    }
)


class FaultTransport:
    """Raises or returns whatever the fault under test requires."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls = 0

    def send(self, prompt, schema, timeout_s) -> TransportResponse:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return TransportResponse(
            text=str(self.outcome),
            prompt_tokens=40,
            completion_tokens=12,
            reasoning_tokens=6,
        )


class TextModelStub:
    def complete_structured(self, prompt, schema, route):
        return {"ranked_parent_asins": ["TINY-A"]}, Usage(calls=1)


def agent_for(outcome: object):
    """A full agent whose generative route fails in the requested way."""

    interpreter = ApiInterpreter(
        FaultTransport(outcome),
        ApiConfig(route=PRIMARY_ROUTE, backoff_base_s=0.0),
        credential=SECRET,
        sleep=lambda _s: None,
    )
    selection = GatewaySelection(
        interpreter=interpreter,
        text_model=TextModelStub(),
        route=PRIMARY_ROUTE,
        degraded=False,
    )
    agent, route_id = build_agent(
        CATALOG,
        RuntimeConfig(enable_llm_reranker=False),
        model_selection=selection,
    )
    return agent, route_id


def assert_contract_valid(case: unittest.TestCase, response: object) -> None:
    """The official turn_response contract, asserted the way the evaluator reads it."""

    case.assertIsInstance(response, dict)
    case.assertIsInstance(response.get("message"), str)

    ask = response.get("ask_attribute")
    case.assertTrue(ask is None or ask in ALLOWED_ATTRIBUTES, f"bad ask_attribute {ask!r}")

    recommendations = response.get("recommendations")
    case.assertIsInstance(recommendations, list)
    case.assertLessEqual(len(recommendations), 10)

    ids = []
    for item in recommendations:
        case.assertIsInstance(item, dict)
        parent_asin = item.get("parent_asin")
        case.assertIsInstance(parent_asin, str)
        case.assertIn(parent_asin, CATALOG_IDS, "recommended an id outside the catalog")
        ids.append(parent_asin)
    case.assertEqual(len(ids), len(set(ids)), "duplicate recommendation")

    # DG-01: the two actions are mutually exclusive.
    if ask is not None:
        case.assertEqual(ids, [], "clarify must carry no recommendations")

    usage = response.get("usage")
    if usage is not None:
        case.assertIsInstance(usage, dict)
        for key in ("prompt_tokens", "completion_tokens"):
            case.assertIsInstance(usage.get(key), int)
            case.assertGreaterEqual(usage[key], 0)


FAULTS = {
    "malformed_json": "not json at all {{",
    "empty_response": "",
    "prose_instead_of_json": "Sure! I think you want boots.",
    "json_array_not_object": "[1, 2, 3]",
    "truncated_json": '{"inferred_mode": "buying", "operations": [',
    "refusal": ModelRefused("I cannot help with that", PRIMARY_ROUTE),
    "timeout": ModelTimeout("provider did not answer", PRIMARY_ROUTE),
    "connection_error": ModelUnavailable("connection reset", PRIMARY_ROUTE),
    "rate_limited": ModelUnavailable("HTTP 429", PRIMARY_ROUTE),
    "server_error": ModelUnavailable("HTTP 503", PRIMARY_ROUTE),
    "unexpected_exception": RuntimeError("component blew up"),
    "operation_flood": json.dumps(
        {
            "inferred_mode": "buying",
            "operations": [
                {
                    "operation": "add",
                    "attribute": "feature",
                    "new_value": f"feature {index}",
                    "polarity": "include",
                    "strength": "soft",
                    "confidence": 0.5,
                }
                for index in range(500)
            ],
        }
    ),
    "hallucinated_attribute": json.dumps(
        {
            "inferred_mode": "buying",
            "operations": [
                {"operation": "add", "attribute": "vibe", "new_value": "cool",
                 "polarity": "include", "strength": "soft", "confidence": 0.9}
            ],
        }
    ),
}


class FaultMatrixTests(unittest.TestCase):
    """Exit gate: zero uncaught exceptions, every response contract-valid."""

    def test_every_fault_still_produces_a_valid_turn(self) -> None:
        for name, outcome in FAULTS.items():
            with self.subTest(fault=name):
                agent, _route = agent_for(outcome)
                self.addCleanup(agent.close)
                agent.reset(f"fault-{name}", {"preference_tags": ["comfort"]})
                response = agent.respond(
                    f"fault-{name}", "I'm looking for cotton shoes.", 1, 10
                )
                assert_contract_valid(self, response)

    def test_a_faulting_route_still_completes_a_whole_session(self) -> None:
        """A mid-session failure must not poison later turns."""

        agent, _route = agent_for(ModelTimeout("slow", PRIMARY_ROUTE))
        self.addCleanup(agent.close)
        agent.reset("long", {})
        for turn in range(1, 11):
            response = agent.respond("long", "I'm looking for cotton shoes.", turn, 10)
            assert_contract_valid(self, response)

    def test_recovery_after_a_transient_failure(self) -> None:
        """One bad turn does not permanently pin the agent to the fallback."""

        transport = FaultTransport(WELL_FORMED)
        interpreter = ApiInterpreter(
            transport,
            ApiConfig(route=PRIMARY_ROUTE, backoff_base_s=0.0),
            credential=SECRET,
            sleep=lambda _s: None,
        )
        selection = GatewaySelection(
            interpreter=interpreter,
            text_model=TextModelStub(),
            route=PRIMARY_ROUTE,
            degraded=False,
        )
        agent, _ = build_agent(
            CATALOG, RuntimeConfig(enable_llm_reranker=False), model_selection=selection
        )
        self.addCleanup(agent.close)
        agent.reset("recover", {})

        transport.outcome = ModelUnavailable("blip", PRIMARY_ROUTE)
        assert_contract_valid(self, agent.respond("recover", "cotton shoes", 1, 10))
        transport.outcome = WELL_FORMED
        assert_contract_valid(self, agent.respond("recover", "cotton shoes", 2, 10))
        state = agent.sessions.get("recover")
        self.assertTrue(state.active_constraints)


class FloodCapTests(unittest.TestCase):
    def test_operation_flood_is_truncated_deterministically(self) -> None:
        payload = FAULTS["operation_flood"]
        first, second = parse(payload), parse(payload)
        self.assertLessEqual(len(first.delta.operations), MAX_OPERATIONS)
        self.assertEqual(first.delta.operations, second.delta.operations)

    def test_flood_cannot_stack_unbounded_constraints_on_one_attribute(self) -> None:
        agent, _ = agent_for(FAULTS["operation_flood"])
        self.addCleanup(agent.close)
        agent.reset("flood", {})
        agent.respond("flood", "I'm looking for shoes.", 1, 10)
        state = agent.sessions.get("flood")
        per_attribute: dict[str, int] = {}
        for constraint in state.active_constraints:
            key = str(constraint.attribute)
            per_attribute[key] = per_attribute.get(key, 0) + 1
        for attribute, count in per_attribute.items():
            self.assertLessEqual(count, 8, f"{attribute} accumulated {count} constraints")


class CredentialSafetyTests(unittest.TestCase):
    def test_missing_credential_never_fails_a_turn(self) -> None:
        """A keyless environment degrades at construction, not mid-session."""

        agent, route_id = build_agent(CATALOG, environ={})
        self.addCleanup(agent.close)
        self.assertEqual(route_id, "heuristic/local")
        agent.reset("keyless", {})
        assert_contract_valid(
            self, agent.respond("keyless", "I'm looking for cotton shoes.", 1, 10)
        )

    def test_scored_runs_refuse_to_degrade_silently(self) -> None:
        with self.assertRaises(CredentialMissing):
            gateway_from_env({}, allow_degraded=False)

    def test_a_credential_in_error_text_is_redacted_from_traces(self) -> None:
        """Defence in depth: a trace is written to disk and pasted into reports.

        Our transport never echoes the credential, but a future provider error
        or a dependency's exception text might, and by then the trace is
        already on disk.
        """

        state = new_session("secret", {})
        for leak in (
            f"auth rejected for {SECRET}",
            f"request failed: Authorization: Bearer {SECRET}",
            f'{{"api_key": "{SECRET}"}}',
        ):
            with self.subTest(leak=leak[:24]):
                trace = capture(
                    state,
                    "I'm looking for shoes.",
                    1,
                    usage=Usage(prompt_tokens=10, calls=1, route=PRIMARY_ROUTE.route_id),
                    route_id=PRIMARY_ROUTE.route_id,
                    failure=leak,
                )
                rendered = json.dumps(trace.to_dict())
                self.assertNotIn(SECRET, rendered)
                self.assertIn(REDACTED, rendered)

    def test_redaction_leaves_ordinary_failure_text_readable(self) -> None:
        self.assertEqual(redact("provider did not answer in time"),
                         "provider did not answer in time")

    def test_http_error_paths_do_not_echo_the_credential(self) -> None:
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                "https://example.test", 401, "Unauthorized", {},
                io.BytesIO(b'{"error":{"message":"Incorrect API key provided"}}'),
            )

        transport = HttpTransport(SECRET, PRIMARY_ROUTE, opener=opener)
        with self.assertRaises(ModelRefused) as caught:
            transport.send("prompt", {}, 5.0)
        rendered = f"{caught.exception!r} {caught.exception}"
        self.assertNotIn(SECRET, rendered)

    def test_socket_timeout_does_not_echo_the_credential(self) -> None:
        def opener(request, timeout=None):
            raise socket.timeout("too slow")

        transport = HttpTransport(SECRET, PRIMARY_ROUTE, opener=opener)
        with self.assertRaises(ModelTimeout) as caught:
            transport.send("prompt", {}, 5.0)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_no_committed_fixture_contains_a_credential(self) -> None:
        """Guards against a real key reaching the repository through a fixture."""

        pattern = re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_-]{16,}")
        root = pathlib.Path(__file__).parent / "fixtures"
        offenders = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
