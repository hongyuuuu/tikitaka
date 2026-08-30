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
from tikitaka.state.reducer import StateReducer
from tikitaka.state.session import SessionState, new_session


CATALOG_SEQUENCE = tuple(f"P-{index:02d}" for index in range(1, 13))
CATALOG_IDS = frozenset(CATALOG_SEQUENCE)
CANDIDATES = [candidate(item, rank, 1.0 / rank) for rank, item in enumerate(CATALOG_SEQUENCE, 1)]


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


def mode_delta(
    mode: InferredMode,
    operations: tuple[StateOperation, ...] = (),
) -> StateDelta:
    return StateDelta(
        inferred_mode=mode,
        mode_confidence=0.9,
        operations=operations,
        generality=0.5,
        rejected_operations=0,
        schema_version=STRUCTURED_OUTPUT_SCHEMA_VERSION,
    )


def no_preference_delta(attribute: Attribute) -> StateDelta:
    operation = StateOperation(
        operation=StateOperationKind.NO_PREFERENCE,
        attribute=attribute,
        old_value=None,
        new_value=None,
        scope=OperationScope.ATTRIBUTE,
        polarity=None,
        strength=None,
        confidence=None,
    )
    return mode_delta(InferredMode.BROWSING, (operation,))


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


def make_owner_integrated_agent(*, interpreter, decision=None):
    sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
    agent = ShoppingAgent(
        sessions=sessions,
        reducer=StateReducer(),
        interpreter=interpreter,
        query_builder=FakeQueryBuilder(),
        retriever=DeterministicRetriever(CANDIDATES, CATALOG_IDS),
        decision_policy=decision or ScriptedDecisionPolicy(default=recommendation()),
        reranker=DeterministicReranker(CATALOG_SEQUENCE),
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

    def test_full_reranker_result_does_not_gain_an_eleventh_fallback_item(self) -> None:
        preferred = [item.parent_asin for item in CANDIDATES[2:12]]
        agent, _ = make_agent(reranker=DeterministicReranker(preferred))
        agent.reset("session", {})

        response = agent.respond("session", "query", 1, 10)
        ids = [item["parent_asin"] for item in response["recommendations"]]

        self.assertEqual(ids, preferred)
        self.assertEqual(len(ids), 10)

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


class RepresentativeTraceTest(unittest.TestCase):
    """P2 exit-gate traces using Person 1's merged state and reducer."""

    def test_buying_trace_accumulates_visible_mode_then_recommends(self) -> None:
        message = "I need blue walking shoes for a purchase this week."
        interpreter = ScriptedInterpreter(script={message: mode_delta(InferredMode.BUYING)})
        agent, sessions = make_owner_integrated_agent(interpreter=interpreter)
        agent.reset("buying", {"summary": "prefers practical products"})

        response = agent.respond("buying", message, 1, 10)

        self.assertEqual(sessions.get("buying").mode, InferredMode.BUYING)
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(len(response["recommendations"]), 10)

    def test_browsing_trace_uses_visible_mode_and_one_clarification(self) -> None:
        message = "I'm exploring ideas and not ready to buy yet."
        interpreter = ScriptedInterpreter(script={message: mode_delta(InferredMode.BROWSING)})
        agent, sessions = make_owner_integrated_agent(
            interpreter=interpreter,
            decision=ScriptedDecisionPolicy(default=clarification()),
        )
        agent.reset("browsing", {})

        response = agent.respond("browsing", message, 1, 10)

        self.assertEqual(sessions.get("browsing").mode, InferredMode.BROWSING)
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [])

    def test_intent_override_trace_increments_version_and_reopens_products(self) -> None:
        interpreter = ScriptedInterpreter(
            script={
                "I want walking shoes.": mode_delta(InferredMode.BUYING),
                "Actually, start over with a daypack.": reset_delta(),
            }
        )
        agent, sessions = make_owner_integrated_agent(interpreter=interpreter)
        agent.reset("override", {})
        first = agent.respond("override", "I want walking shoes.", 1, 10)
        first_ids = {item["parent_asin"] for item in first["recommendations"]}

        second = agent.respond("override", "Actually, start over with a daypack.", 2, 10)
        second_ids = {item["parent_asin"] for item in second["recommendations"]}
        state = sessions.get("override")

        self.assertEqual(state.intent_version, 2)
        self.assertEqual(state.shown_product_ids, frozenset(second_ids))
        self.assertEqual(first_ids, second_ids)

    def test_boundary_trace_records_no_preference_and_does_not_repeat_question(self) -> None:
        interpreter = ScriptedInterpreter(
            script={
                "I'm still exploring.": mode_delta(InferredMode.BROWSING),
                "I have no material preference.": no_preference_delta(Attribute.MATERIAL),
            }
        )
        policy = ScriptedDecisionPolicy(
            decisions={1: clarification(), 2: recommendation()},
            default=recommendation(),
        )
        agent, sessions = make_owner_integrated_agent(interpreter=interpreter, decision=policy)
        agent.reset("boundary", {})

        first = agent.respond("boundary", "I'm still exploring.", 1, 10)
        second = agent.respond("boundary", "I have no material preference.", 2, 10)
        state = sessions.get("boundary")

        self.assertEqual(first["ask_attribute"], "material")
        self.assertIn(Attribute.MATERIAL, state.no_preference)
        self.assertIsNone(second["ask_attribute"])
        self.assertGreater(len(second["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main()
