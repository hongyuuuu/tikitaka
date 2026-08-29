from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tikitaka.decision import ResponsePolicy
from tikitaka.orchestration.runtime import (
    DeterministicRuntimeConfig,
    VisibleMessageInterpreter,
    build_deterministic_agent,
)
from tikitaka.ranking import DeterministicRanker
from tikitaka.retrieval import SparseStructuredRetriever
from tikitaka.state.query_builder import ActiveQueryBuilder
from tikitaka.state.reducer import StateReducer
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


if __name__ == "__main__":
    unittest.main()
