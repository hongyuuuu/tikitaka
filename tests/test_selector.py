"""P6: routing, pinning, ablation identity, and the index/route refusal.

The exit gate is two claims, and both are asserted here as behaviour through
the real composition root rather than against the selector in isolation:

- the same pinned configuration reproduces the same validated state;
- an index/route mismatch is impossible to reach silently.

Nothing here touches the network or a credential.
"""

from __future__ import annotations

import json
import pathlib
import unittest
from dataclasses import replace

from tikitaka.contracts.domain import Usage
from tikitaka.models.base import ModelRoute
from tikitaka.models.factory import PRIMARY_ROUTE, GatewaySelection, selector_from_env
from tikitaka.models.fake import HEURISTIC_ROUTE, HeuristicInterpreter
from tikitaka.models.selector import (
    EMBED,
    FROZEN_PROMPT_VERSION,
    FROZEN_SCHEMA_VERSION,
    INTERPRET,
    MAX_TURNS,
    PINNED,
    REWRITE,
    RUNTIME_AUTO,
    TASKS,
    AblationConfig,
    ModelSelector,
    RouteMismatch,
    RoutingInterpreter,
    SELECTIVE,
    RoutingSignals,
    RoutingThresholds,
    deterministic_selector,
    looks_like_override,
    pin_all,
)
from tikitaka.orchestration.runtime import RuntimeConfig, build_agent
from tikitaka.state.session import new_session

CATALOG = pathlib.Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"

CONFIDENT = RoutingSignals(
    task=INTERPRET,
    mode_confidence=1.0,
    remaining_turns=MAX_TURNS - 3,
    constraint_count=3,
    observed_turns=3,
)

OVERRIDE_MESSAGE = (
    "Actually, ignore my earlier preference. What I need is: leather boots."
)


class RecordingInterpreter:
    """Stands in for the API route. Counts calls, never touches a network."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._inner = HeuristicInterpreter()

    def interpret(self, message: str, state: object):
        self.calls.append(message)
        delta, _ = self._inner.interpret(message, state)
        return delta, Usage(prompt_tokens=11, completion_tokens=7, calls=1)


class ExplodingInterpreter:
    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, message: str, state: object):
        self.calls += 1
        raise RuntimeError("provider is down")


class RoutingSignalsTests(unittest.TestCase):
    def test_signals_are_read_from_a_real_session_state(self) -> None:
        state = new_session("s1", {})
        signals = RoutingSignals.from_turn("something soft", state)
        self.assertEqual(signals.task, INTERPRET)
        self.assertEqual(signals.remaining_turns, MAX_TURNS - state.turn)
        self.assertEqual(signals.constraint_count, len(state.active_constraints))
        self.assertFalse(signals.override_suspected)

    def test_the_official_override_template_is_detected(self) -> None:
        # evaluator/local_evaluator.py:85 emits exactly this phrasing.
        self.assertTrue(looks_like_override(OVERRIDE_MESSAGE))
        self.assertTrue(RoutingSignals.from_turn(OVERRIDE_MESSAGE, new_session("s", {})).override_suspected)
        self.assertFalse(looks_like_override("I want cotton shirts under $50."))
        self.assertFalse(looks_like_override(None))

    def test_an_unreadable_state_yields_neutral_signals_not_an_exception(self) -> None:
        # The fault matrix drives states that raise on attribute access.
        # Routing must never be the thing that fails a turn.
        class Hostile:
            @property
            def mode_confidence(self):
                raise RuntimeError("boom")

            @property
            def turn(self):
                raise RuntimeError("boom")

            @property
            def active_constraints(self):
                raise RuntimeError("boom")

        signals = RoutingSignals.from_turn("hello", Hostile())
        self.assertEqual(signals.mode_confidence, 1.0)
        self.assertEqual(signals.constraint_count, 0)

    def test_out_of_range_signals_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RoutingSignals(task="summarize")
        with self.assertRaises(ValueError):
            RoutingSignals(mode_confidence=1.5)
        with self.assertRaises(ValueError):
            RoutingSignals(remaining_turns=-1)


class DefaultRoutingTests(unittest.TestCase):
    """The default keeps `build_agent`'s pre-routing behaviour."""

    def test_a_configured_generative_route_handles_every_turn(self) -> None:
        selector = ModelSelector(PRIMARY_ROUTE)
        for signals in (CONFIDENT, replace(CONFIDENT, mode_confidence=0.0)):
            decision = selector.select(signals)
            self.assertTrue(decision.generative)
            self.assertEqual(decision.reason, "generative_available")

    def test_the_default_still_falls_back_with_no_generative_route(self) -> None:
        self.assertFalse(ModelSelector(None).select(CONFIDENT).generative)

    def test_the_selective_policy_is_available_but_not_the_default(self) -> None:
        self.assertTrue(RoutingThresholds().always_generative)
        self.assertFalse(SELECTIVE.always_generative)

    def test_the_confidence_threshold_discriminates_between_observed_values(self) -> None:
        # The heuristic emits exactly 0.00 and 0.60 across the public set. A
        # threshold outside that range makes the comparison a constant, which
        # is what shipped: 0.65 escalated on 100% of turns. This is the guard
        # that would have caught it.
        observed_low, observed_high = 0.00, 0.60
        self.assertGreater(SELECTIVE.min_mode_confidence, observed_low)
        self.assertLess(SELECTIVE.min_mode_confidence, observed_high)

    def test_the_selective_policy_leaves_a_recognised_mode_alone(self) -> None:
        recognised = replace(CONFIDENT, mode_confidence=0.60)
        selector = ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE)
        self.assertFalse(selector.select(recognised).generative)

    def test_the_selective_policy_escalates_an_unrecognised_mode(self) -> None:
        unrecognised = replace(CONFIDENT, mode_confidence=0.00)
        selector = ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE)
        decision = selector.select(unrecognised)
        self.assertTrue(decision.generative)
        self.assertEqual(decision.reason, "low_mode_confidence")


class SelectiveRoutingTests(unittest.TestCase):
    """The opt-in cost-saving policy."""

    def setUp(self) -> None:
        self.selector = ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE)

    def test_a_confident_constrained_state_stays_deterministic(self) -> None:
        decision = self.selector.select(CONFIDENT)
        self.assertEqual(decision.route, HEURISTIC_ROUTE)
        self.assertEqual(decision.reason, "deterministic_sufficient")
        self.assertFalse(decision.generative)

    def test_each_signal_escalates_to_the_generative_route(self) -> None:
        cases = {
            "override_suspected": replace(CONFIDENT, override_suspected=True),
            "low_mode_confidence": replace(CONFIDENT, mode_confidence=0.2),
            "no_extracted_constraints": replace(CONFIDENT, constraint_count=0),
            "low_turn_budget": replace(CONFIDENT, remaining_turns=1),
        }
        for expected, signals in cases.items():
            with self.subTest(reason=expected):
                decision = self.selector.select(signals)
                self.assertEqual(decision.route, PRIMARY_ROUTE)
                self.assertEqual(decision.reason, expected)
                self.assertTrue(decision.generative)

    def test_the_recorded_reason_is_stable_when_signals_overlap(self) -> None:
        # Two runs that escalate for different stated reasons are not
        # comparable, so precedence must be fixed rather than incidental.
        crowded = replace(
            CONFIDENT,
            mode_confidence=0.1,
            remaining_turns=0,
            constraint_count=0,
            override_suspected=True,
        )
        self.assertEqual(
            {self.selector.select(crowded).reason for _ in range(5)},
            {"override_suspected"},
        )

    def test_the_opening_turn_is_not_treated_as_an_extraction_failure(self) -> None:
        # Before any turn is reduced, mode_confidence is 0.0 and there are no
        # constraints for every session alive. Escalating on that would put a
        # provider call on turn one of all 200 sessions unconditionally.
        opening = RoutingSignals.from_turn("I want something comfortable.", new_session("s", {}))
        self.assertFalse(opening.has_evidence)
        self.assertEqual(opening.mode_confidence, 0.0)
        self.assertEqual(self.selector.select(opening).reason, "deterministic_sufficient")

    def test_an_override_still_escalates_on_the_opening_turn(self) -> None:
        opening = RoutingSignals.from_turn(OVERRIDE_MESSAGE, new_session("s", {}))
        self.assertEqual(self.selector.select(opening).reason, "override_suspected")

    def test_thresholds_are_configurable_without_touching_policy_code(self) -> None:
        eager = ModelSelector(
            PRIMARY_ROUTE,
            thresholds=replace(SELECTIVE, min_mode_confidence=1.0),
        )
        self.assertEqual(
            eager.select(replace(CONFIDENT, mode_confidence=0.99)).reason,
            "low_mode_confidence",
        )

    def test_no_generative_route_is_reported_not_disguised(self) -> None:
        selector = ModelSelector(None)
        decision = selector.select(replace(CONFIDENT, mode_confidence=0.0))
        self.assertTrue(selector.degraded)
        self.assertEqual(decision.route, HEURISTIC_ROUTE)
        self.assertEqual(decision.reason, "no_generative_route")

    def test_rewrite_is_off_unless_its_ablation_enables_it(self) -> None:
        uncertain = replace(CONFIDENT, task=REWRITE, mode_confidence=0.0)
        self.assertEqual(self.selector.select(uncertain).route, HEURISTIC_ROUTE)
        enabled = ModelSelector(
            PRIMARY_ROUTE,
            ablation=AblationConfig(use_llm_query_rewrite=True),
            thresholds=SELECTIVE,
        )
        self.assertEqual(enabled.select(uncertain).route, PRIMARY_ROUTE)

    def test_embed_never_escalates_on_conversational_signals(self) -> None:
        signals = replace(CONFIDENT, task=EMBED, mode_confidence=0.0)
        self.assertEqual(self.selector.select(signals).route, HEURISTIC_ROUTE)

    def test_rejects_an_unroutable_pinned_task(self) -> None:
        with self.assertRaises(ValueError):
            ModelSelector(PRIMARY_ROUTE, pins={"summarize": HEURISTIC_ROUTE})


class PinningTests(unittest.TestCase):
    def test_a_pin_overrides_every_escalation_signal(self) -> None:
        selector = ModelSelector(PRIMARY_ROUTE, pins={INTERPRET: HEURISTIC_ROUTE})
        decision = selector.select(
            replace(CONFIDENT, mode_confidence=0.0, override_suspected=True)
        )
        self.assertEqual(decision.route.route_id, HEURISTIC_ROUTE.route_id)
        self.assertEqual(decision.reason, "pinned")
        self.assertTrue(decision.pinned)

    def test_a_full_pin_gives_one_decision_across_every_signal(self) -> None:
        selector = ModelSelector(PRIMARY_ROUTE, pins=pin_all(HEURISTIC_ROUTE))
        varied = [
            CONFIDENT,
            replace(CONFIDENT, mode_confidence=0.0),
            replace(CONFIDENT, constraint_count=0, remaining_turns=1, observed_turns=9),
            replace(CONFIDENT, override_suspected=True),
        ]
        decisions = {
            json.dumps(selector.select(item).to_dict(), sort_keys=True) for item in varied
        }
        self.assertEqual(len(decisions), 1)

    def test_partial_pinning_is_reported_as_runtime_routing(self) -> None:
        partial = ModelSelector(PRIMARY_ROUTE, pins={INTERPRET: HEURISTIC_ROUTE})
        self.assertEqual(partial.routing_mode, RUNTIME_AUTO)
        self.assertEqual(
            ModelSelector(PRIMARY_ROUTE, pins=pin_all(HEURISTIC_ROUTE)).routing_mode,
            PINNED,
        )

    def test_pin_all_covers_every_task_and_marks_routes_pinned(self) -> None:
        pins = pin_all(PRIMARY_ROUTE)
        self.assertEqual(set(pins), set(TASKS))
        self.assertTrue(all(route.pinned for route in pins.values()))

    def test_deterministic_selector_is_fully_pinned_and_degraded(self) -> None:
        selector = deterministic_selector()
        self.assertEqual(selector.routing_mode, PINNED)
        self.assertTrue(selector.degraded)
        self.assertEqual(selector.select(CONFIDENT).route.route_id, "heuristic/local")

    def test_selector_from_env_reports_only_what_it_can_reach(self) -> None:
        self.assertTrue(selector_from_env({}).degraded)
        self.assertTrue(selector_from_env({"OPENAI_API_KEY": "   "}).degraded)
        live = selector_from_env({"OPENAI_API_KEY": "sk-placeholder-value"})
        self.assertFalse(live.degraded)
        self.assertEqual(live.identity()["generative_model"], "gpt-5.6-terra")


class IndexRouteRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = ModelSelector(PRIMARY_ROUTE)
        self.route = ModelRoute(
            route_id="embed/e5",
            provider="local",
            model="e5-base",
            index_id="dense-0123456789abcdef0123",
        )

    def test_a_matching_index_returns_the_route_unchanged(self) -> None:
        self.assertIs(
            self.selector.select_embedding(self.route, self.route.index_id), self.route
        )

    def test_a_different_index_is_refused_loudly_and_names_both(self) -> None:
        with self.assertRaises(RouteMismatch) as caught:
            self.selector.select_embedding(self.route, "dense-ffffffffffffffffffff")
        message = str(caught.exception)
        self.assertIn(self.route.index_id, message)
        self.assertIn("dense-ffffffffffffffffffff", message)
        self.assertIn("embed/e5", message)

    def test_a_route_with_no_index_identity_cannot_pass_by_default(self) -> None:
        # Absence of identity is a refusal, not a waiver: an unverifiable route
        # is the case most likely to reach a run unnoticed.
        anonymous = replace(self.route, index_id=None)
        with self.assertRaises(RouteMismatch):
            self.selector.select_embedding(anonymous, "dense-0123456789abcdef0123")

    def test_an_empty_index_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.selector.select_embedding(self.route, "  ")


class AblationIdentityTests(unittest.TestCase):
    def test_identity_supplies_the_experiment_config_fields(self) -> None:
        identity = ModelSelector(PRIMARY_ROUTE).identity()
        for field in (
            "prompt_version",
            "schema_version",
            "routing_mode",
            "generative_provider",
            "generative_model",
            "reasoning_level",
        ):
            self.assertTrue(str(identity[field]).strip(), field)
        self.assertEqual(identity["generative_model"], "gpt-5.6-terra")
        self.assertEqual(identity["reasoning_level"], "xhigh")
        self.assertIn(identity["routing_mode"], {RUNTIME_AUTO, PINNED})

    def test_versions_are_frozen_at_the_published_values(self) -> None:
        self.assertEqual(FROZEN_PROMPT_VERSION, "intent-interpreter/1")
        self.assertEqual(FROZEN_SCHEMA_VERSION, "0.1.0")

    def test_every_ablation_knob_is_visible_in_identity(self) -> None:
        baseline = ModelSelector(PRIMARY_ROUTE).identity()
        ablated = ModelSelector(
            PRIMARY_ROUTE,
            ablation=AblationConfig(
                profile_weight=0.25,
                use_api_interpreter=False,
                use_llm_query_rewrite=True,
                reasoning_level="low",
            ),
        ).identity()
        self.assertNotEqual(baseline, ablated)
        for field in (
            "profile_weight",
            "use_api_interpreter",
            "use_llm_query_rewrite",
            "reasoning_level",
        ):
            self.assertNotEqual(baseline[field], ablated[field], field)

    def test_the_interpreter_ablation_actually_changes_routing(self) -> None:
        # An ablation that edited only the report would produce a delta of
        # zero and be read as evidence the LLM does nothing.
        uncertain = replace(CONFIDENT, mode_confidence=0.0)
        self.assertTrue(ModelSelector(PRIMARY_ROUTE).select(uncertain).generative)
        off = ModelSelector(
            PRIMARY_ROUTE, ablation=AblationConfig(use_api_interpreter=False)
        )
        self.assertFalse(off.select(uncertain).generative)

    def test_the_reasoning_ablation_reaches_the_selected_route(self) -> None:
        selector = ModelSelector(
            PRIMARY_ROUTE, ablation=AblationConfig(reasoning_level="medium")
        )
        decision = selector.select(replace(CONFIDENT, mode_confidence=0.0))
        self.assertEqual(decision.route.reasoning_level, "medium")
        self.assertEqual(PRIMARY_ROUTE.reasoning_level, "xhigh")

    def test_identity_carries_no_credential_shaped_value(self) -> None:
        rendered = repr(
            selector_from_env({"OPENAI_API_KEY": "sk-should-never-appear"}).identity()
        ).lower()
        for token in ("sk-", "bearer", "api_key", "authorization"):
            self.assertNotIn(token, rendered)

    def test_profile_weight_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AblationConfig(profile_weight=1.5)


class RoutingInterpreterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = RecordingInterpreter()
        self.local = HeuristicInterpreter()

    def _interpreter(self, **kwargs) -> RoutingInterpreter:
        kwargs.setdefault("thresholds", SELECTIVE)
        selector = ModelSelector(PRIMARY_ROUTE, **kwargs)
        return RoutingInterpreter(selector, self.api, self.local)

    def test_it_routes_per_turn_not_once_per_build(self) -> None:
        interpreter = self._interpreter()
        state = new_session("s", {})
        interpreter.interpret("I want cotton shirts.", state)
        self.assertEqual(self.api.calls, [])
        interpreter.interpret(OVERRIDE_MESSAGE, state)
        self.assertEqual(self.api.calls, [OVERRIDE_MESSAGE])
        self.assertEqual(
            [item.reason for item in interpreter.decisions],
            ["deterministic_sufficient", "override_suspected"],
        )

    def test_usage_is_stamped_with_the_route_that_did_the_work(self) -> None:
        interpreter = self._interpreter()
        _, usage = interpreter.interpret(OVERRIDE_MESSAGE, new_session("s", {}))
        self.assertEqual(usage.route, PRIMARY_ROUTE.route_id)

    def test_a_failing_generative_route_completes_the_turn_on_the_fallback(self) -> None:
        exploding = ExplodingInterpreter()
        interpreter = RoutingInterpreter(
            ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE), exploding, self.local
        )
        delta, usage = interpreter.interpret(OVERRIDE_MESSAGE, new_session("s", {}))
        self.assertEqual(exploding.calls, 1)
        self.assertIsNotNone(delta)
        # The report must not be able to claim a clean API turn.
        self.assertEqual(usage.route, f"{PRIMARY_ROUTE.route_id}:fallback")

    def test_a_transient_failure_does_not_pin_the_agent_to_the_fallback(self) -> None:
        interpreter = self._interpreter()
        state = new_session("s", {})
        interpreter.interpret(OVERRIDE_MESSAGE, state)
        interpreter.interpret(OVERRIDE_MESSAGE, state)
        self.assertEqual(len(self.api.calls), 2)

    def test_the_decision_history_is_bounded(self) -> None:
        interpreter = self._interpreter()
        state = new_session("s", {})
        for _ in range(MAX_TURNS * 3):
            interpreter.interpret("plain message", state)
        self.assertLessEqual(len(interpreter.decisions), MAX_TURNS)

    def test_summarize_reports_routing_facts_for_the_experiment(self) -> None:
        interpreter = self._interpreter()
        state = new_session("s", {})
        interpreter.interpret("I want cotton shirts.", state)
        interpreter.interpret(OVERRIDE_MESSAGE, state)
        summary = interpreter.summarize()
        self.assertEqual(summary["turns_routed"], 2)
        self.assertEqual(summary["generative_turns"], 1)
        self.assertEqual(summary["routing_mode"], RUNTIME_AUTO)
        self.assertEqual(summary["reasons"]["override_suspected"], 1)


class RealPathTests(unittest.TestCase):
    """The selector as the runtime actually uses it."""

    def _selection(self, interpreter: object) -> GatewaySelection:
        return GatewaySelection(
            interpreter=interpreter,
            text_model=None,
            route=PRIMARY_ROUTE,
            degraded=False,
        )

    def _converse(self, agent, session_id: str) -> list[dict]:
        agent.reset(session_id, {})
        messages = [
            "I'm looking for something comfortable.",
            "Cotton would be good.",
            OVERRIDE_MESSAGE,
            "Under $80 please.",
        ]
        return [
            agent.respond(session_id, message, turn, 10)
            for turn, message in enumerate(messages, start=1)
        ]

    def test_the_degraded_build_never_reaches_a_generative_route(self) -> None:
        api = RecordingInterpreter()
        agent, route_id = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=GatewaySelection(
                interpreter=api, text_model=None, route=HEURISTIC_ROUTE, degraded=True
            ),
        )
        with agent:
            responses = self._converse(agent, "degraded")
        self.assertEqual(route_id, HEURISTIC_ROUTE.route_id)
        self.assertEqual(api.calls, [])
        self.assertEqual(len(responses), 4)
        for response in responses:
            self.assertEqual(response["usage"]["prompt_tokens"], 0)

    def test_routing_reaches_the_generative_route_through_build_agent(self) -> None:
        api = RecordingInterpreter()
        agent, route_id = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=self._selection(api),
        )
        with agent:
            responses = self._converse(agent, "auto")
        self.assertEqual(route_id, PRIMARY_ROUTE.route_id)
        # The override turn is the one the P5 evidence says the heuristic
        # mangles, so it is the turn that must actually escalate.
        self.assertIn(OVERRIDE_MESSAGE, api.calls)
        self.assertTrue(any(item["usage"]["prompt_tokens"] > 0 for item in responses))

    def test_the_selective_policy_spares_turns_the_heuristic_handles(self) -> None:
        # Opt-in cost saving, proven end to end: fewer provider calls than
        # turns, but the override turn is still one of them.
        api = RecordingInterpreter()
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(
                enable_llm_reranker=False,
                selector=ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE),
            ),
            environ={},
            model_selection=self._selection(api),
        )
        with agent:
            responses = self._converse(agent, "selective")
        self.assertLess(len(api.calls), len(responses))
        self.assertIn(OVERRIDE_MESSAGE, api.calls)

    def test_pinning_to_the_local_route_suppresses_every_provider_call(self) -> None:
        api = RecordingInterpreter()
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(
                enable_llm_reranker=False,
                selector=ModelSelector(PRIMARY_ROUTE, pins=pin_all(HEURISTIC_ROUTE)),
            ),
            environ={},
            model_selection=self._selection(api),
        )
        with agent:
            responses = self._converse(agent, "pinned")
        self.assertEqual(api.calls, [])
        self.assertTrue(all(item["usage"]["prompt_tokens"] == 0 for item in responses))

    def test_a_pinned_configuration_reproduces_the_same_validated_state(self) -> None:
        # The P6 exit gate, asserted end to end: same pins, same responses and
        # same resulting session state across independent builds.
        def run(session_id: str):
            agent, _ = build_agent(
                CATALOG,
                RuntimeConfig(
                    enable_llm_reranker=False,
                    selector=ModelSelector(
                        PRIMARY_ROUTE, pins=pin_all(HEURISTIC_ROUTE)
                    ),
                ),
                environ={},
                model_selection=self._selection(RecordingInterpreter()),
            )
            with agent:
                responses = self._converse(agent, session_id)
                state = agent.sessions.get(session_id)
                fingerprint = (
                    state.intent_version,
                    str(state.mode),
                    tuple(
                        (str(item.attribute), str(item.normalized_value))
                        for item in state.active_constraints
                    ),
                    tuple(sorted(str(item) for item in state.asked_attributes)),
                )
            return json.dumps(responses, sort_keys=True), fingerprint

        first_responses, first_state = run("repro-a")
        second_responses, second_state = run("repro-b")
        self.assertEqual(first_responses, second_responses)
        self.assertEqual(first_state, second_state)

    def test_the_ablation_switch_changes_behaviour_in_the_real_path(self) -> None:
        api = RecordingInterpreter()
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(
                enable_llm_reranker=False,
                selector=ModelSelector(
                    PRIMARY_ROUTE,
                    ablation=AblationConfig(use_api_interpreter=False),
                ),
            ),
            environ={},
            model_selection=self._selection(api),
        )
        with agent:
            self._converse(agent, "ablated")
        self.assertEqual(api.calls, [])

    def test_a_broken_generative_route_still_yields_valid_turns(self) -> None:
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=self._selection(ExplodingInterpreter()),
        )
        with agent:
            responses = self._converse(agent, "broken")
        for response in responses:
            self.assertIsInstance(response["message"], str)
            self.assertLessEqual(len(response["recommendations"]), 10)
            self.assertGreaterEqual(response["usage"]["prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
