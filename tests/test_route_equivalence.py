"""P4 exit gate: the fake and API routes must agree on validated state.

Per `docs/PERSON_1_BUILD_PLAN.md`, P4 is done when a synthetic multi-turn
conversation produces the same validated state through both adapters. Any
divergence on the templated turns is a prompt or schema defect, not a model
quality difference, because the fixtures author the model reply to express the
same operations the deterministic route derives.

Confidence is excluded from the comparison. It is a model judgement that may
legitimately differ between routes; what must not differ is which constraints
end up active, with which polarity and strength.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tikitaka.models.api_llm import ApiConfig, ApiInterpreter, TransportResponse
from tikitaka.models.base import ModelRoute
from tikitaka.models.factory import (
    PRIMARY_ROUTE,
    describe_route,
    gateway_from_env,
    interpreter_from_env,
)
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.state.extractor import Extractor
from tikitaka.state.session import new_session
from tikitaka.state.trace import FORBIDDEN_KEYS, capture, summarize, write_jsonl

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "conversations"
SECRET = "sk-not-a-real-key-0123456789"

PROFILE = {"preference_tags": ["fit", "comfort"], "summary": "prior purchases"}


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ScriptedTransport:
    """Returns the fixture's authored model reply, turn by turn."""

    def __init__(self, replies) -> None:
        self.replies = [json.dumps(item) for item in replies]
        self.index = 0

    def send(self, prompt, schema, timeout_s) -> TransportResponse:
        reply = self.replies[min(self.index, len(self.replies) - 1)]
        self.index += 1
        return TransportResponse(
            text=reply, prompt_tokens=520, completion_tokens=140, reasoning_tokens=70
        )


def api_extractor(fixture: dict) -> Extractor:
    transport = ScriptedTransport([turn["model_response"] for turn in fixture["turns"]])
    interpreter = ApiInterpreter(
        transport,
        ApiConfig(route=PRIMARY_ROUTE, backoff_base_s=0.0),
        credential=SECRET,
        sleep=lambda _s: None,
    )
    return Extractor(interpreter=interpreter)


def snapshot(state) -> dict:
    """Everything that must match, minus model-judgement confidence."""

    return {
        "mode": str(state.mode),
        "intent_version": state.intent_version,
        "constraints": sorted(
            (
                str(c.attribute),
                str(c.normalized_value),
                str(c.polarity),
                str(c.strength),
            )
            for c in state.active_constraints
        ),
        "revalidation": sorted(str(c.attribute) for c in state.revalidation_constraints),
        "no_preference": sorted(str(item) for item in state.no_preference),
        "exhausted": sorted(str(item) for item in state.exhausted_attributes),
    }


def run(extractor: Extractor, fixture: dict, session_id: str):
    state = new_session(session_id, PROFILE)
    traces = []
    for index, turn in enumerate(fixture["turns"], start=1):
        result = extractor.ingest(state, turn["message"], index)
        traces.append(
            capture(
                state,
                turn["message"],
                index,
                usage=result.usage,
                route_id="test",
                used_fallback=result.used_fallback,
                failure=result.failure,
            )
        )
    return state, traces


class RouteEquivalenceTests(unittest.TestCase):
    """The P4 exit gate."""

    def test_fake_and_api_agree_on_templated_turns(self) -> None:
        for name in ("buying.json", "browsing.json", "boundary.json"):
            with self.subTest(scenario=name):
                fixture = load(name)
                heuristic_state, _ = run(
                    Extractor(interpreter=HeuristicInterpreter()), fixture, "h"
                )
                api_state, _ = run(api_extractor(fixture), fixture, "a")
                self.assertEqual(snapshot(heuristic_state), snapshot(api_state))

    def test_boundary_answer_is_no_preference_on_both_routes(self) -> None:
        fixture = load("boundary.json")
        for extractor, label in (
            (Extractor(interpreter=HeuristicInterpreter()), "heuristic"),
            (api_extractor(fixture), "api"),
        ):
            with self.subTest(route=label):
                state, _ = run(extractor, fixture, label)
                self.assertIn("material", state.no_preference)
                self.assertNotIn("material", state.exhausted_attributes)

    def test_api_route_recovers_an_override_the_heuristic_mangles(self) -> None:
        """Where the routes legitimately diverge, and why the LLM is worth it.

        The override message carries no "I'm looking for" template, so the
        deterministic route cannot tell which attribute changed and shreds the
        remainder into a bogus constraint. The API route replaces the category
        and lets DG-03 preserve the budget.
        """

        fixture = load("intent_override.json")
        heuristic_state, _ = run(
            Extractor(interpreter=HeuristicInterpreter()), fixture, "h"
        )
        api_state, _ = run(api_extractor(fixture), fixture, "a")

        api = {
            str(c.attribute): str(c.normalized_value)
            for c in api_state.active_constraints
        }
        self.assertEqual(api.get("category"), "wool sweaters")
        self.assertEqual(api.get("budget"), "120.0")
        self.assertEqual(api_state.intent_version, 2)

        heuristic = {
            str(c.attribute): str(c.normalized_value)
            for c in heuristic_state.active_constraints
        }
        self.assertNotEqual(heuristic.get("category"), "wool sweaters")


class TraceTests(unittest.TestCase):
    def test_traces_capture_state_and_cost_without_labels(self) -> None:
        fixture = load("buying.json")
        state, traces = run(api_extractor(fixture), fixture, "trace")

        self.assertEqual(len(traces), 2)
        rendered = json.dumps([item.to_dict() for item in traces])
        for forbidden in FORBIDDEN_KEYS:
            self.assertNotIn(forbidden, rendered)
        self.assertNotIn(SECRET, rendered)

        totals = summarize(traces)
        self.assertEqual(totals["turns"], 2)
        self.assertEqual(totals["calls"], 2)
        self.assertEqual(totals["prompt_tokens"], 1040)
        self.assertEqual(totals["reasoning_tokens"], 140)
        self.assertEqual(totals["fallback_turns"], 0)

    def test_traces_round_trip_as_jsonl(self) -> None:
        fixture = load("browsing.json")
        _, traces = run(api_extractor(fixture), fixture, "jsonl")
        with tempfile.TemporaryDirectory() as directory:
            path = write_jsonl(pathlib.Path(directory) / "t.jsonl", traces)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["turn"], 1)


class FactoryTests(unittest.TestCase):
    def test_missing_credential_degrades_to_the_heuristic_route(self) -> None:
        interpreter, route_id = interpreter_from_env({})
        self.assertIsInstance(interpreter, HeuristicInterpreter)
        self.assertEqual(route_id, "heuristic/local")

    def test_evaluation_runs_can_refuse_to_degrade_silently(self) -> None:
        from tikitaka.models.base import CredentialMissing

        with self.assertRaises(CredentialMissing):
            interpreter_from_env({}, allow_degraded=False)

    def test_present_credential_selects_the_primary_route(self) -> None:
        interpreter, route_id = interpreter_from_env({"OPENAI_API_KEY": SECRET})
        self.assertIsInstance(interpreter, ApiInterpreter)
        self.assertEqual(route_id, "primary/gpt-5.6-terra")
        self.assertNotIn(SECRET, repr(interpreter))

        gateway = gateway_from_env({"OPENAI_API_KEY": SECRET})
        self.assertFalse(gateway.degraded)
        self.assertIsNotNone(gateway.text_model)
        self.assertEqual(gateway.route, PRIMARY_ROUTE)
        self.assertNotIn(SECRET, repr(gateway.text_model))

    def test_describe_route_reports_without_leaking_the_credential(self) -> None:
        described = describe_route({"OPENAI_API_KEY": SECRET})
        self.assertTrue(described["credential_present"])
        self.assertFalse(described["degraded"])
        self.assertEqual(described["reasoning_level"], "medium")
        self.assertNotIn(SECRET, json.dumps(described))

        degraded = describe_route({})
        self.assertTrue(degraded["degraded"])
        self.assertEqual(degraded["route_id"], "heuristic/local")


if __name__ == "__main__":
    unittest.main()
