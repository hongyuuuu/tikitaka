"""P4 adapter tests, driven entirely by a transport double.

No network, no credential, no cost. Everything below the `Transport` seam is
provider-specific and lands with the real endpoint; everything above it is
tested here.
"""

from __future__ import annotations

import json
import unittest

from tikitaka.contracts.domain import Usage
from tikitaka.models.api_llm import (
    PROMPT_VERSION,
    ApiConfig,
    ApiInterpreter,
    ApiTextModel,
    InMemoryResponseCache,
    TransportResponse,
    build_prompt,
    credential_from_env,
)
from tikitaka.models.base import (
    CredentialMissing,
    MalformedModelOutput,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
)
from tikitaka.state.extractor import Extractor
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import make_delta, operation
from tikitaka.state.session import new_session

ROUTE = ModelRoute(
    route_id="primary/gpt-5.6-terra",
    provider="openai",
    model="gpt-5.6-terra",
    reasoning_level="xhigh",
)
SECRET = "sk-not-a-real-key-0123456789"

GOOD = json.dumps(
    {
        "inferred_mode": "buying",
        "mode_confidence": 0.9,
        "generality": 0.2,
        "operations": [
            {
                "operation": "add",
                "attribute": "material",
                "new_value": "waterproof leather",
                "polarity": "include",
                "strength": "hard",
                "confidence": 0.9,
            }
        ],
    }
)


class FakeTransport:
    """Returns scripted responses and records what it was asked to send."""

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def send(self, prompt, schema, timeout_s) -> TransportResponse:
        self.prompts.append(prompt)
        item = self.script.pop(0) if self.script else self.script
        if isinstance(item, Exception):
            raise item
        return TransportResponse(
            text=str(item),
            prompt_tokens=100,
            # 40 of the 60 output tokens were reasoning, matching how the
            # provider actually reports the breakdown.
            completion_tokens=60,
            reasoning_tokens=40,
        )


def build(transport, **kwargs) -> ApiInterpreter:
    config = ApiConfig(route=ROUTE, backoff_base_s=0.0, **kwargs)
    return ApiInterpreter(
        transport, config, credential=SECRET, sleep=lambda _seconds: None
    )


class HappyPathTests(unittest.TestCase):
    def test_well_formed_response_produces_delta_and_usage(self) -> None:
        transport = FakeTransport(GOOD)
        delta, usage = build(transport).interpret("I need waterproof boots", None)

        self.assertEqual(delta.inferred_mode, "buying")
        self.assertEqual(len(delta.operations), 1)
        self.assertEqual(delta.operations[0].attribute, "material")
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.repairs, 0)
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 60)
        self.assertEqual(usage.reasoning_tokens, 40)
        self.assertEqual(usage.model, "gpt-5.6-terra")
        self.assertEqual(usage.reasoning_level, "xhigh")
        self.assertGreaterEqual(usage.latency_ms, 0.0)

    def test_structured_text_model_supports_non_state_schemas(self) -> None:
        transport = FakeTransport('{"ranked_parent_asins":["B","A"]}')
        model = ApiTextModel(transport, ROUTE)

        output, usage = model.complete_structured(
            "rank these",
            {
                "type": "object",
                "properties": {"ranked_parent_asins": {"type": "array"}},
                "required": ["ranked_parent_asins"],
                "additionalProperties": False,
            },
            ROUTE,
        )

        self.assertEqual(output["ranked_parent_asins"], ["B", "A"])
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.route, ROUTE.route_id)

    def test_structured_text_model_preserves_usage_on_malformed_json(self) -> None:
        model = ApiTextModel(FakeTransport("not-json"), ROUTE)
        with self.assertRaises(MalformedModelOutput) as caught:
            model.complete_structured("rank", {"type": "object"}, ROUTE)
        self.assertEqual(caught.exception.usage.calls, 1)
        self.assertEqual(caught.exception.usage.completion_tokens, 60)

    def test_hallucinated_attribute_is_dropped_not_trusted(self) -> None:
        payload = json.dumps(
            {
                "inferred_mode": "buying",
                "operations": [
                    {"operation": "add", "attribute": "vibe", "new_value": "cool",
                     "polarity": "include", "strength": "soft", "confidence": 0.9},
                    {"operation": "add", "attribute": "color", "new_value": "red",
                     "polarity": "include", "strength": "soft", "confidence": 0.9},
                ],
            }
        )
        delta, _ = build(FakeTransport(payload)).interpret("something", None)
        self.assertEqual(len(delta.operations), 1)
        self.assertEqual(delta.operations[0].attribute, "color")
        self.assertEqual(delta.rejected_operations, 1)

    def test_cost_is_estimated_from_configured_rates(self) -> None:
        interpreter = build(
            FakeTransport(GOOD),
            prompt_cost_per_1k=1.0,
            completion_cost_per_1k=2.0,
        )
        _, usage = interpreter.interpret("hi", None)
        # 100 prompt tokens at 1.0/1k, 60 completion tokens at 2.0/1k. The 40
        # reasoning tokens are already inside the 60 and are not priced again.
        self.assertAlmostEqual(usage.estimated_cost, 0.1 + 0.12)
        self.assertEqual(usage.cost_currency, "USD")


class RepairTests(unittest.TestCase):
    def test_malformed_output_triggers_exactly_one_repair(self) -> None:
        transport = FakeTransport("not json at all {{", GOOD)
        delta, usage = build(transport).interpret("boots", None)

        self.assertEqual(len(transport.prompts), 2)
        self.assertEqual(usage.calls, 2)
        self.assertEqual(usage.repairs, 1)
        self.assertEqual(len(delta.operations), 1)

    def test_repair_prompt_shows_the_model_its_bad_output(self) -> None:
        transport = FakeTransport("garbage {{", GOOD)
        build(transport).interpret("boots", None)
        self.assertIn("garbage {{", transport.prompts[1])
        self.assertIn("corrected JSON", transport.prompts[1])

    def test_persistent_malformed_output_raises_with_usage_attached(self) -> None:
        transport = FakeTransport("bad", "still bad")
        with self.assertRaises(MalformedModelOutput) as caught:
            build(transport).interpret("boots", None)
        self.assertEqual(len(transport.prompts), 2)
        self.assertEqual(caught.exception.usage.calls, 2)
        self.assertEqual(caught.exception.usage.repairs, 1)

    def test_repairs_can_be_disabled_by_configuration(self) -> None:
        transport = FakeTransport("bad", GOOD)
        with self.assertRaises(MalformedModelOutput):
            build(transport, max_repairs=0).interpret("boots", None)
        self.assertEqual(len(transport.prompts), 1)


class TransportFailureTests(unittest.TestCase):
    def test_transient_unavailability_is_retried_then_succeeds(self) -> None:
        transport = FakeTransport(ModelUnavailable("rate limited", ROUTE), GOOD)
        delta, usage = build(transport).interpret("boots", None)
        self.assertEqual(len(transport.prompts), 2)
        self.assertEqual(usage.calls, 1)
        self.assertEqual(len(delta.operations), 1)

    def test_exhausted_attempts_raise_unavailable(self) -> None:
        transport = FakeTransport(
            ModelUnavailable("down", ROUTE), ModelUnavailable("down", ROUTE)
        )
        with self.assertRaises(ModelUnavailable):
            build(transport).interpret("boots", None)

    def test_timeout_is_not_retried(self) -> None:
        transport = FakeTransport(ModelTimeout("slow", ROUTE), GOOD)
        with self.assertRaises(ModelTimeout):
            build(transport).interpret("boots", None)
        self.assertEqual(len(transport.prompts), 1)


class CredentialTests(unittest.TestCase):
    def test_missing_credential_fails_at_construction(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(CredentialMissing):
                    ApiInterpreter(
                        FakeTransport(GOOD), ApiConfig(route=ROUTE), credential=value
                    )

    def test_credential_never_appears_in_repr_or_prompt(self) -> None:
        transport = FakeTransport(GOOD)
        interpreter = build(transport)
        interpreter.interpret("boots", None)
        self.assertNotIn(SECRET, repr(interpreter))
        self.assertNotIn(SECRET, str(vars(interpreter)))
        for prompt in transport.prompts:
            self.assertNotIn(SECRET, prompt)

    def test_credential_from_env_reads_by_name_only(self) -> None:
        self.assertEqual(
            credential_from_env("SOME_KEY", {"SOME_KEY": SECRET}), SECRET
        )
        with self.assertRaises(CredentialMissing):
            credential_from_env("SOME_KEY", {})
        with self.assertRaises(CredentialMissing):
            credential_from_env("SOME_KEY", {"SOME_KEY": "  "})


class CacheTests(unittest.TestCase):
    def test_cache_hit_adds_no_call_or_cost(self) -> None:
        cache = InMemoryResponseCache()
        transport = FakeTransport(GOOD)
        interpreter = build(transport, cache=cache, prompt_cost_per_1k=1.0)

        first_delta, first_usage = interpreter.interpret("boots", None)
        second_delta, second_usage = interpreter.interpret("boots", None)

        self.assertEqual(len(transport.prompts), 1)
        self.assertEqual(first_delta, second_delta)
        self.assertGreater(first_usage.calls, 0)
        self.assertEqual(second_usage.calls, 0)
        self.assertEqual(second_usage.prompt_tokens, 0)
        self.assertTrue(second_usage.cache_hit)
        self.assertIsNone(second_usage.estimated_cost)

    def test_cache_is_off_by_default(self) -> None:
        transport = FakeTransport(GOOD, GOOD)
        interpreter = build(transport)
        interpreter.interpret("boots", None)
        interpreter.interpret("boots", None)
        self.assertEqual(len(transport.prompts), 2)


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = new_session("p", {"preference_tags": ["fit"]})
        StateReducer().apply(
            self.state,
            make_delta(
                inferred_mode="buying",
                operations=(
                    operation(
                        "add",
                        attribute="category",
                        new_value="hiking boots",
                        polarity="include",
                        strength="hard",
                        confidence=0.9,
                    ),
                ),
            ),
            1,
        )

    def test_prompt_is_a_pure_function_of_message_and_state(self) -> None:
        first = build_prompt("need boots", self.state)
        second = build_prompt("need boots", self.state)
        self.assertEqual(first, second)
        self.assertIn(PROMPT_VERSION, first)

    def test_prompt_carries_active_state_not_the_transcript(self) -> None:
        prompt = build_prompt("need boots", self.state)
        self.assertIn("hiking boots", prompt)
        self.assertIn("intent_version: 1", prompt)
        self.assertIn("need boots", prompt)

    def test_prompt_teaches_the_boundary_versus_exhaustion_distinction(self) -> None:
        prompt = build_prompt("x", self.state)
        self.assertIn("please use your judgment", prompt)
        self.assertIn("additional preference", prompt)

    def test_prompt_never_contains_catalog_products(self) -> None:
        prompt = build_prompt("need boots", self.state)
        self.assertNotIn("parent_asin", prompt)
        self.assertNotIn("B0", prompt)


class DegradationTests(unittest.TestCase):
    """The extractor turns an API failure into a route change, not an outage."""

    def test_persistent_malformed_output_degrades_and_keeps_spent_usage(self) -> None:
        state = new_session("d", {})
        interpreter = build(FakeTransport("bad", "worse"))
        result = Extractor(interpreter=interpreter).ingest(
            state,
            "I'm looking for boots. A key requirement is: waterproof leather.",
            1,
        )
        self.assertTrue(result.used_fallback)
        self.assertTrue(state.active_constraints)
        # Two real calls were paid for before falling back.
        self.assertEqual(result.usage.calls, 2)

    def test_unavailable_provider_degrades_without_raising(self) -> None:
        state = new_session("d2", {})
        interpreter = build(
            FakeTransport(
                ModelUnavailable("down", ROUTE), ModelUnavailable("down", ROUTE)
            )
        )
        result = Extractor(interpreter=interpreter).ingest(
            state, "I'm looking for sweaters, but I'm still exploring.", 1
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(state.mode, "browsing")
        self.assertIsInstance(result.usage, Usage)


if __name__ == "__main__":
    unittest.main()
