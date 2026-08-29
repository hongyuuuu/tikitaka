from __future__ import annotations

import unittest

from tests.fakes.components import (
    DeterministicReranker,
    DeterministicRetriever,
    FakeQueryBuilder,
    MalformedInterpreter,
    MalformedReranker,
    RaisingDecisionPolicy,
    RaisingInterpreter,
    RaisingReranker,
    RaisingRetriever,
    ScriptedDecisionPolicy,
    ScriptedInterpreter,
    candidate,
    empty_delta,
)
from tikitaka.config import STRUCTURED_OUTPUT_SCHEMA_VERSION
from tikitaka.contracts import (
    Attribute,
    DecisionReasonCode,
    InferredMode,
    OperationScope,
    StateDelta,
    StateOperation,
    StateOperationKind,
    TurnAction,
    TurnDecision,
)
from tikitaka.orchestration.scaffold import ScaffoldReducer, ScaffoldState
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent


CATALOG_IDS = frozenset(f"P-{index:02d}" for index in range(1, 13))
CANDIDATES = [candidate(item, rank, 1.0 / rank) for rank, item in enumerate(CATALOG_IDS, 1)]


def recommendation() -> TurnDecision:
    return TurnDecision(
        action=TurnAction.RECOMMEND,
        ask_attribute=None,
        reason_code=DecisionReasonCode.RANKING_STABLE,
        reason="test recommendation",
        expected_information_gain=0.0,
    )


def clarification() -> TurnDecision:
    return TurnDecision(
        action=TurnAction.CLARIFY,
        ask_attribute=Attribute.MATERIAL,
        reason_code=DecisionReasonCode.VALUABLE_CLARIFICATION,
        reason="material changes ranking",
        expected_information_gain=0.8,
    )


def reset_delta() -> StateDelta:
    return StateDelta(
        inferred_mode=InferredMode.BUYING,
        mode_confidence=1.0,
        operations=(StateOperation(
            operation=StateOperationKind.RESET,
            attribute=None,
            old_value=None,
            new_value=None,
            scope=OperationScope.INTENT,
            polarity=None,
            strength=None,
            confidence=None,
        ),),
        generality=0.0,
        rejected_operations=0,
        schema_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
    )


def make_agent(*, interpreter=None, retriever=None, decision=None, reranker=None):
    sessions = SessionRegistry(
        lambda session_id, profile: ScaffoldState(session_id=session_id, profile_seed=profile)
    )
    agent = ShoppingAgent(
        sessions=sessions,
        reducer=ScaffoldReducer(),
        interpreter=interpreter or ScriptedInterpreter(),
        query_builder=FakeQueryBuilder(),
        retriever=retriever or DeterministicRetriever(CANDIDATES, CATALOG_IDS),
        decision_policy=decision or ScriptedDecisionPolicy(default=recommendation()),
        reranker=reranker or DeterministicReranker([item.parent_asin for item in CANDIDATES]),
        catalog_ids=CATALOG_IDS,
    )
    return agent, sessions


class OrchestrationTest(unittest.TestCase):
    def test_reset_isolates_sessions_and_defensively_copies_profile(self) -> None:
        agent, sessions = make_agent()
        profile = {"preference_tags": ["blue"]}
        agent.reset("one", profile)
        agent.reset("two", {"summary": "other"})
        profile["preference_tags"].append("mutated")

        self.assertEqual(sessions.profile_snapshot("one")["preference_tags"], ["blue"])
        self.assertNotEqual(sessions.get("one"), sessions.get("two"))

        agent.reset("one", {"summary": "replacement"})
        self.assertEqual(sessions.profile_snapshot("one")["summary"], "replacement")
        self.assertEqual(sessions.usage_events("one"), ())

    def test_calls_before_reset_and_invalid_requests_are_safe(self) -> None:
        agent, _ = make_agent()
        for arguments in (
            ("missing", "hello", 1, 10),
            ("", "hello", 1, 10),
            ("missing", "hello", 0, 10),
            ("missing", "hello", 11, 10),
            ("missing", "hello", 1, 5),
        ):
            response = agent.respond(*arguments)
            self.assertEqual(response["ask_attribute"], None)
            self.assertEqual(response["recommendations"], [])
            self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_clarify_is_mutually_exclusive_and_final_turn_forces_recommendation(self) -> None:
        policy = ScriptedDecisionPolicy(default=clarification())
        agent, sessions = make_agent(decision=policy)
        agent.reset("session", {})

        response = agent.respond("session", "vague", 1, 10)
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [])
        self.assertIn(Attribute.MATERIAL, sessions.get("session").asked_attributes)

        response = agent.respond("session", "still vague", 10, 10)
        self.assertIsNone(response["ask_attribute"])
        self.assertGreater(len(response["recommendations"]), 0)

    def test_reranker_output_is_unique_shortlist_valid_and_limited_to_ten(self) -> None:
        preferred = [CANDIDATES[0].parent_asin, "NOT-IN-CATALOG", CANDIDATES[0].parent_asin]
        agent, _ = make_agent(reranker=DeterministicReranker(preferred))
        agent.reset("session", {})
        response = agent.respond("session", "query", 1, 10)
        ids = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= CATALOG_IDS)

    def test_malformed_and_raising_components_produce_valid_fallbacks(self) -> None:
        cases = (
            {"interpreter": MalformedInterpreter()},
            {"interpreter": RaisingInterpreter()},
            {"retriever": RaisingRetriever()},
            {"decision": RaisingDecisionPolicy()},
            {"reranker": MalformedReranker()},
            {"reranker": RaisingReranker()},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                agent, _ = make_agent(**overrides)
                session_id = f"session-{index}"
                agent.reset(session_id, {})
                response = agent.respond(session_id, "query", 1, 10)
                self.assertIsInstance(response["message"], str)
                self.assertIsNone(response["ask_attribute"])
                self.assertIsInstance(response["recommendations"], list)
                self.assertGreaterEqual(response["usage"]["prompt_tokens"], 0)

    def test_usage_is_attributable_and_reports_only_current_turn(self) -> None:
        agent, sessions = make_agent()
        agent.reset("session", {})
        first = agent.respond("session", "one", 1, 10)
        second = agent.respond("session", "two", 2, 10)
        self.assertEqual(first["usage"], {"prompt_tokens": 14, "completion_tokens": 6})
        self.assertEqual(second["usage"], first["usage"])
        self.assertEqual(
            [event.component for event in sessions.usage_events("session")],
            ["interpreter", "reranker", "interpreter", "reranker"],
        )

    def test_new_intent_version_has_distinct_shown_history(self) -> None:
        interpreter = ScriptedInterpreter(script={"restart": reset_delta()}, default=empty_delta())
        agent, sessions = make_agent(interpreter=interpreter)
        agent.reset("session", {})
        first = agent.respond("session", "first", 1, 10)
        first_ids = {item["parent_asin"] for item in first["recommendations"]}
        self.assertEqual(sessions.get("session").intent_version, 1)

        second = agent.respond("session", "restart", 2, 10)
        second_ids = {item["parent_asin"] for item in second["recommendations"]}
        state = sessions.get("session")
        self.assertEqual(state.intent_version, 2)
        self.assertEqual(state.shown_product_ids, frozenset(second_ids))
        self.assertEqual(first_ids, second_ids)

    def test_component_boundary_contains_no_hidden_evaluator_fields(self) -> None:
        class SpyInterpreter(ScriptedInterpreter):
            seen = None

            def interpret(self, message, state):
                self.seen = (message, state)
                return super().interpret(message, state)

        spy = SpyInterpreter()
        agent, _ = make_agent(interpreter=spy)
        agent.reset("session", {"summary": "visible profile"})
        agent.respond("session", "visible message", 1, 10)
        message, state = spy.seen
        self.assertEqual(message, "visible message")
        self.assertFalse(hasattr(state, "ground_truth"))
        self.assertFalse(hasattr(state, "scenario_type"))
        self.assertFalse(hasattr(state, "intent_card"))


if __name__ == "__main__":
    unittest.main()
