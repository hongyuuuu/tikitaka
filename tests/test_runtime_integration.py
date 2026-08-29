from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tikitaka.decision import ResponsePolicy
from tikitaka.decision import ResponsePolicyConfig
from tikitaka.contracts import Usage
from tikitaka.models.base import ModelTimeout
from tikitaka.models.factory import GatewaySelection, PRIMARY_ROUTE
from tikitaka.orchestration.runtime import (
    DeterministicRuntimeConfig,
    RuntimeConfig,
    VisibleMessageInterpreter,
    build_agent,
    build_deterministic_agent,
)
from tikitaka.ranking import DeterministicRanker
from tikitaka.retrieval import SparseStructuredRetriever
from tikitaka.state.query_builder import ActiveQueryBuilder
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import make_delta, operation
from tikitaka.state.session import SessionState


CATALOG = Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"
CATALOG_IDS = frozenset({"TINY-A", "TINY-B", "TINY-C"})


class RuntimeIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = build_deterministic_agent(CATALOG)
        self.addCleanup(self.agent.close)

    def test_composition_root_integrates_all_owner_boundaries(self) -> None:
        self.assertIsInstance(self.agent.sessions.get("missing"), type(None))
        self.assertIsInstance(self.agent._reducer, StateReducer)
        self.assertIsInstance(self.agent._interpreter, VisibleMessageInterpreter)
        self.assertIsInstance(self.agent._query_builder, ActiveQueryBuilder)
        self.assertIsInstance(self.agent._retriever, SparseStructuredRetriever)
        self.assertIsInstance(self.agent._decision_policy, ResponsePolicy)
        self.assertIsInstance(self.agent._reranker, DeterministicRanker)

    def test_free_form_visible_message_uses_owner_retrieval_and_ranking(self) -> None:
        self.agent.reset("free-form", {"preference_tags": ["untrusted profile hint"]})

        response = self.agent.respond(
            "free-form", "blue cotton walking shoe", 1, 10
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "TINY-A"}])
        self.assertIsNone(response["ask_attribute"])
        state = self.agent.sessions.get("free-form")
        self.assertIsInstance(state, SessionState)
        self.assertEqual(
            [(str(item.attribute), item.normalized_value) for item in state.active_constraints],
            [("other", "blue cotton walking shoe")],
        )
        self.assertFalse(hasattr(state, "ground_truth"))
        self.assertFalse(hasattr(state, "scenario_type"))

    def test_interleaved_sessions_keep_state_and_results_isolated(self) -> None:
        self.agent.reset("shoes", {"preference_tags": ["comfort"]})
        self.agent.reset("bags", {"preference_tags": ["lightweight"]})

        with ThreadPoolExecutor(max_workers=2) as executor:
            shoe_future = executor.submit(
                self.agent.respond, "shoes", "blue cotton walking shoe", 1, 10
            )
            bag_future = executor.submit(
                self.agent.respond, "bags", "red canvas daypack", 1, 10
            )
            shoe_response = shoe_future.result()
            bag_response = bag_future.result()

        self.assertEqual(shoe_response["recommendations"][0]["parent_asin"], "TINY-A")
        self.assertEqual(bag_response["recommendations"][0]["parent_asin"], "TINY-C")
        shoe_state = self.agent.sessions.get("shoes")
        bag_state = self.agent.sessions.get("bags")
        self.assertNotEqual(shoe_state.active_constraints, bag_state.active_constraints)
        self.assertEqual(shoe_state.shown_product_ids, frozenset({"TINY-A"}))
        self.assertEqual(bag_state.shown_product_ids, frozenset({"TINY-C"}))
        self.assertTrue(shoe_state.shown_product_ids <= CATALOG_IDS)
        self.assertTrue(bag_state.shown_product_ids <= CATALOG_IDS)

    def test_runtime_config_rejects_invalid_integration_values(self) -> None:
        with self.assertRaises(ValueError):
            DeterministicRuntimeConfig(candidate_limit=0)
        with self.assertRaises(ValueError):
            DeterministicRuntimeConfig(profile_weight=1.1)

    def test_visible_override_replaces_only_corrected_attribute_and_reopens_ids(self) -> None:
        self.agent.reset("override", {})
        first = self.agent.respond(
            "override",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )
        before = self.agent.sessions.get("override")
        self.assertEqual(before.intent_version, 1)
        self.assertEqual(first["recommendations"][0]["parent_asin"], "TINY-A")

        second = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: leather.",
            2,
            10,
        )
        after = self.agent.sessions.get("override")

        self.assertEqual(after.intent_version, 2)
        self.assertEqual(
            {str(item.attribute): item.normalized_value for item in after.active_constraints},
            {"category": "shoes", "material": "leather"},
        )
        self.assertTrue(
            all(item["parent_asin"] in CATALOG_IDS for item in second["recommendations"])
        )


class PrimaryRuntimeIntegrationTest(unittest.TestCase):
    class Interpreter:
        def interpret(self, message, state):
            return make_delta(
                inferred_mode="buying",
                mode_confidence=0.9,
                generality=0.2,
                operations=(operation(
                    "add",
                    attribute="other",
                    new_value="Tiny Store",
                    polarity="include",
                    strength="soft",
                    confidence=0.8,
                ),),
            ), Usage(
                prompt_tokens=5,
                completion_tokens=2,
                calls=1,
                provider="fake",
                model=PRIMARY_ROUTE.model,
                reasoning_level=PRIMARY_ROUTE.reasoning_level,
                route=PRIMARY_ROUTE.route_id,
            )

    class TextModel:
        def __init__(self):
            self.calls = 0

        def complete_structured(self, prompt, schema, route):
            self.calls += 1
            return {"ranked_parent_asins": ["TINY-C", "TINY-A", "TINY-B"]}, Usage(
                prompt_tokens=7,
                completion_tokens=3,
                calls=1,
                provider="fake",
                model=route.model,
                reasoning_level=route.reasoning_level,
                route=route.route_id,
            )

    def test_primary_interpreter_and_llm_reranker_are_selected_together(self) -> None:
        text_model = self.TextModel()
        selection = GatewaySelection(
            interpreter=self.Interpreter(),
            text_model=text_model,
            route=PRIMARY_ROUTE,
            degraded=False,
        )
        agent, _route = build_agent(
            CATALOG,
            RuntimeConfig(
                decision=ResponsePolicyConfig(generality_threshold=1.0),
            ),
            model_selection=selection,
        )
        self.addCleanup(agent.close)
        agent.reset("primary", {})

        response = agent.respond("primary", "find something", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "TINY-C")
        self.assertEqual(response["usage"], {"prompt_tokens": 12, "completion_tokens": 5})
        self.assertEqual(text_model.calls, 1)
        self.assertEqual(agent.runtime_route_id, PRIMARY_ROUTE.route_id)
        self.assertFalse(agent.degraded)

    def test_primary_failure_degrades_for_the_current_turn(self) -> None:
        class FailingInterpreter:
            def interpret(self, message, state):
                error = ModelTimeout("slow", PRIMARY_ROUTE)
                error.usage = Usage(
                    prompt_tokens=9,
                    completion_tokens=1,
                    calls=1,
                    provider="fake",
                    model=PRIMARY_ROUTE.model,
                    reasoning_level=PRIMARY_ROUTE.reasoning_level,
                    route=PRIMARY_ROUTE.route_id,
                )
                raise error

        selection = GatewaySelection(
            interpreter=FailingInterpreter(),
            text_model=self.TextModel(),
            route=PRIMARY_ROUTE,
            degraded=False,
        )
        agent, _route = build_agent(CATALOG, model_selection=selection)
        self.addCleanup(agent.close)
        agent.reset("fallback", {})

        response = agent.respond("fallback", "blue cotton walking shoe", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "TINY-A")
        self.assertEqual(response["usage"]["prompt_tokens"], 9)
        event = agent.sessions.usage_events("fallback")[0]
        self.assertEqual(event.usage.route, f"{PRIMARY_ROUTE.route_id}:fallback")

    def test_experiment_can_pin_deterministic_reranking_on_primary_intent(self) -> None:
        text_model = self.TextModel()
        selection = GatewaySelection(
            interpreter=self.Interpreter(),
            text_model=text_model,
            route=PRIMARY_ROUTE,
            degraded=False,
        )
        agent, _route = build_agent(
            CATALOG,
            RuntimeConfig(
                decision=ResponsePolicyConfig(generality_threshold=1.0),
                enable_llm_reranker=False,
            ),
            model_selection=selection,
        )
        self.addCleanup(agent.close)
        agent.reset("pinned", {})

        response = agent.respond("pinned", "find something", 1, 10)

        self.assertEqual(text_model.calls, 0)
        self.assertEqual(response["usage"], {"prompt_tokens": 5, "completion_tokens": 2})

    def test_primary_llm_pin_rejects_a_missing_text_model(self) -> None:
        selection = GatewaySelection(
            interpreter=self.Interpreter(),
            text_model=None,
            route=PRIMARY_ROUTE,
            degraded=False,
        )
        with self.assertRaisesRegex(ValueError, "structured text model"):
            build_agent(CATALOG, model_selection=selection)

    def test_missing_credential_keeps_offline_default_deterministic(self) -> None:
        agent, _route = build_agent(CATALOG, environ={})
        self.addCleanup(agent.close)
        agent.reset("offline", {})

        response = agent.respond("offline", "red canvas daypack", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "TINY-C")
        self.assertEqual(agent.runtime_route_id, "heuristic/local")
        self.assertTrue(agent.degraded)


if __name__ == "__main__":
    unittest.main()
