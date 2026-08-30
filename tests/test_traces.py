"""Traces as an artifact: the routing record, the query summary, and the labels.

`tikitaka/state/trace.py` had machinery and tests since P5 but no producer, so
two defects survived until `scripts/capture_traces.py` actually wrote a file:
the query summary was always empty, and the routing reason had nowhere to go.
Both are covered here.

Nothing in this module touches the network or a credential.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from scripts.capture_traces import SCENARIOS, capture_environment
from tikitaka.contracts.domain import Usage
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import make_delta, operation
from tikitaka.state.session import new_session
from tikitaka.state.trace import (
    FORBIDDEN_KEYS,
    REDACTED,
    capture,
    describe_query,
    summarize,
)


def _state_with_constraints():
    state = new_session("trace-session", {"segment": "outdoors"})
    reducer = StateReducer()
    delta = make_delta(
        inferred_mode="buying",
        mode_confidence=0.9,
        operations=(
            operation(
                "add",
                attribute="material",
                new_value="leather",
                polarity="include",
                strength="hard",
                confidence=0.9,
            ),
            operation(
                "exclude",
                attribute="color",
                new_value="pink",
                polarity="exclude",
                strength="soft",
                confidence=0.6,
            ),
            operation("no_preference", attribute="brand"),
        ),
    )
    return reducer.apply(state, delta, 1)


class QuerySummaryTests(unittest.TestCase):
    def test_the_summary_is_populated_not_empty(self) -> None:
        # The regression that shipped: `active_query_summary` was a field
        # nothing ever wrote, so every trace carried "" where the query goes.
        trace = capture(_state_with_constraints(), "leather, not pink", 1)
        self.assertTrue(trace.query_summary.strip())

    def test_it_renders_polarity_strength_and_no_preference(self) -> None:
        summary = describe_query(_state_with_constraints())
        self.assertIn("material!=leather", summary)
        self.assertIn("-color=pink", summary)
        self.assertIn("~brand", summary)

    def test_it_is_stable_for_the_same_state(self) -> None:
        state = _state_with_constraints()
        self.assertEqual(describe_query(state), describe_query(state))

    def test_an_empty_state_summarizes_to_an_empty_string(self) -> None:
        self.assertEqual(describe_query(new_session("s", {})), "")


class RoutingRecordTests(unittest.TestCase):
    def test_the_routing_reason_and_mode_are_recorded(self) -> None:
        trace = capture(
            _state_with_constraints(),
            "leather please",
            2,
            route_id="primary/gpt-5.6-terra",
            route_reason="override_suspected",
            routing_mode="runtime_auto",
        )
        self.assertEqual(trace.route_reason, "override_suspected")
        self.assertEqual(trace.routing_mode, "runtime_auto")

    def test_summarize_aggregates_reasons_across_a_session(self) -> None:
        state = _state_with_constraints()
        traces = [
            capture(state, "m", 1, route_reason="deterministic_sufficient",
                    routing_mode="runtime_auto"),
            capture(state, "m", 2, route_reason="override_suspected",
                    routing_mode="runtime_auto"),
            capture(state, "m", 3, route_reason="deterministic_sufficient",
                    routing_mode="runtime_auto"),
        ]
        summary = summarize(traces)
        self.assertEqual(
            summary["route_reasons"],
            {"deterministic_sufficient": 2, "override_suspected": 1},
        )
        self.assertEqual(summary["routing_mode"], "runtime_auto")

    def test_a_session_with_no_routing_record_summarizes_cleanly(self) -> None:
        state = _state_with_constraints()
        summary = summarize([capture(state, "m", 1)])
        self.assertEqual(summary["route_reasons"], {})
        self.assertEqual(summary["routing_mode"], "")


class TraceSafetyTests(unittest.TestCase):
    def test_capture_is_offline_even_when_the_shell_has_a_credential(self) -> None:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-not-a-real-key-0123456789"
        try:
            self.assertEqual(capture_environment(False), {})
            self.assertIsNone(capture_environment(True))
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous

    def test_no_evaluator_label_can_appear_in_a_trace(self) -> None:
        rendered = json.dumps(
            capture(_state_with_constraints(), "leather", 1).to_dict(), sort_keys=True
        )
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(f'"{key}"', rendered)

    def test_a_credential_in_failure_text_is_redacted(self) -> None:
        trace = capture(
            _state_with_constraints(),
            "leather",
            1,
            failure="401 from provider: api_key=sk-not-a-real-key-0123456789",
        )
        self.assertIn(REDACTED, trace.failure)
        self.assertNotIn("sk-not-a-real-key", trace.failure)

    def test_usage_is_carried_verbatim(self) -> None:
        trace = capture(
            _state_with_constraints(),
            "leather",
            1,
            usage=Usage(prompt_tokens=31, completion_tokens=12, calls=1),
        )
        self.assertEqual(trace.prompt_tokens, 31)
        self.assertEqual(trace.completion_tokens, 12)


class CommittedDemoTraceTests(unittest.TestCase):
    def test_all_four_label_free_offline_demo_traces_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1] / "artifacts" / "traces"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["scenarios"]), set(SCENARIOS))
        self.assertFalse(manifest["route"]["credential_present"])
        self.assertEqual(manifest["route"]["route_id"], "heuristic/local")
        for scenario in SCENARIOS:
            record = manifest["scenarios"][scenario]
            path = root / record["trace_file"]
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), record["turns"])
            self.assertGreater(len(rows), 0)
            self.assertLessEqual(len(rows), 10)
            self.assertIsInstance(record["hit"], bool)
            self.assertEqual(record["hit"], record["first_hit_turn"] is not None)
            self.assertTrue(all(row["route_id"] == "heuristic/local" for row in rows))
            self.assertTrue(all(row["calls"] == 0 for row in rows))
            rendered = json.dumps(rows, sort_keys=True)
            for key in FORBIDDEN_KEYS:
                self.assertNotIn(f'"{key}"', rendered)
        self.assertEqual(
            manifest["scenarios"]["intent_override"]["final_intent_version"], 2
        )


class ExhaustionRecordTests(unittest.TestCase):
    """A spent question must reach the state, not just the interpreter."""

    def _interpreter(self):
        from tikitaka.models.fake import HeuristicInterpreter
        from tikitaka.models.selector import ModelSelector, RoutingInterpreter

        local = HeuristicInterpreter()
        return RoutingInterpreter(ModelSelector(None), local, local)

    def test_a_spent_question_is_recorded_on_the_state(self) -> None:
        # `exhausted_attributes` was empty in every captured trace: the module
        # that noted it, state/extractor.py, is not in the running agent.
        state = new_session("s", {})
        self._interpreter().interpret(
            "I don't have an additional preference for style.", state
        )
        self.assertIn("style", state.exhausted_attributes)

    def test_it_appears_in_the_trace(self) -> None:
        state = new_session("s", {})
        self._interpreter().interpret(
            "I don't have an additional preference for brand.", state
        )
        trace = capture(state, "I don't have an additional preference for brand.", 2)
        self.assertIn("brand", trace.exhausted_attributes)

    def test_exhaustion_stays_distinct_from_no_preference(self) -> None:
        # Conflating them would discard a real Boundary answer.
        state = new_session("s", {})
        self._interpreter().interpret(
            "I don't have an additional preference for style.", state
        )
        self.assertIn("style", state.exhausted_attributes)
        self.assertNotIn("style", state.no_preference)

    def test_an_ordinary_message_records_nothing(self) -> None:
        state = new_session("s", {})
        self._interpreter().interpret("I want leather boots.", state)
        self.assertEqual(state.exhausted_attributes, frozenset())

    def test_a_state_without_the_field_is_tolerated(self) -> None:
        class Bare:
            turn = 1
            mode_confidence = 0.5
            active_constraints = ()

        interpreter = self._interpreter()
        # Must not raise: routing is never allowed to fail a turn.
        interpreter._note_exhaustion(
            "I don't have an additional preference for style.", Bare()
        )


class NoInformationPredicateTests(unittest.TestCase):
    def test_both_official_no_information_templates_are_recognised(self) -> None:
        from tikitaka.models.fake import carries_no_new_constraint

        self.assertTrue(
            carries_no_new_constraint("I don't have an additional preference for style.")
        )
        self.assertTrue(
            carries_no_new_constraint(
                "I don't have a preference for color; please use your judgment."
            )
        )

    def test_a_real_preference_is_not_mistaken_for_silence(self) -> None:
        from tikitaka.models.fake import carries_no_new_constraint

        self.assertFalse(carries_no_new_constraint("For that, what matters is: fabric."))
        self.assertFalse(carries_no_new_constraint("I want leather boots."))
        self.assertFalse(carries_no_new_constraint(""))


class CostDisclosureTests(unittest.TestCase):
    """M6 requires a cost disclosure; a zero rate makes one impossible."""

    def test_default_rates_are_configured(self) -> None:
        from tikitaka.models.api_llm import ApiConfig
        from tikitaka.models.factory import PRIMARY_ROUTE

        config = ApiConfig(route=PRIMARY_ROUTE)
        self.assertGreater(config.prompt_cost_per_1k, 0.0)
        self.assertGreater(config.completion_cost_per_1k, 0.0)
        self.assertEqual(config.cost_currency, "USD")

    def test_a_priced_call_reports_a_non_zero_cost(self) -> None:
        from tikitaka.models.api_llm import ApiConfig
        from tikitaka.models.factory import PRIMARY_ROUTE
        from tikitaka.models.usage import for_route

        config = ApiConfig(route=PRIMARY_ROUTE)
        usage = for_route(
            PRIMARY_ROUTE,
            prompt_tokens=1000,
            completion_tokens=1000,
            prompt_cost_per_1k=config.prompt_cost_per_1k,
            completion_cost_per_1k=config.completion_cost_per_1k,
            cost_currency=config.cost_currency,
        )
        self.assertIsNotNone(usage.estimated_cost)
        self.assertGreater(usage.estimated_cost, 0.0)


class ConsolidationTests(unittest.TestCase):
    """One implementation behind both entry points."""

    def test_the_override_vocabulary_has_a_single_definition(self) -> None:
        # Three copies existed with three different alternation sets, so the
        # same message could be an override to one component and not another.
        from tikitaka.models import fake, selector

        self.assertIs(selector.looks_like_override, fake.looks_like_override)

    def test_the_union_covers_every_previous_variant(self) -> None:
        from tikitaka.models.fake import looks_like_override

        for phrase in (
            "Actually, ignore my earlier preference. What I need is: leather.",
            "forget the blue one",          # only models/fake.py had this
            "what I need is a wool coat",   # only models/selector.py had this
            "on second thought, wool",
            "changed my mind",
        ):
            self.assertTrue(looks_like_override(phrase), phrase)
        self.assertFalse(looks_like_override("I want leather boots."))

    def test_both_ingestion_paths_produce_the_same_state(self) -> None:
        # They used to disagree: the extractor went through the reducer, which
        # also marks the attribute asked, while the live path added to
        # `_exhausted` directly and did not.
        from tikitaka.models.fake import HeuristicInterpreter
        from tikitaka.models.selector import ModelSelector, RoutingInterpreter
        from tikitaka.state.extractor import Extractor

        message = "I don't have an additional preference for brand."

        live = new_session("live", {})
        local = HeuristicInterpreter()
        RoutingInterpreter(ModelSelector(None), local, local).interpret(message, live)

        harness = new_session("harness", {})
        Extractor(interpreter=HeuristicInterpreter()).ingest(harness, message, 1)

        self.assertEqual(live.exhausted_attributes, harness.exhausted_attributes)
        self.assertEqual(live.asked_attributes, harness.asked_attributes)
        self.assertEqual(live.no_preference, harness.no_preference)

    def test_a_spent_question_is_marked_asked_as_well_as_exhausted(self) -> None:
        from tikitaka.models.fake import HeuristicInterpreter
        from tikitaka.models.selector import ModelSelector, RoutingInterpreter

        state = new_session("s", {})
        local = HeuristicInterpreter()
        RoutingInterpreter(ModelSelector(None), local, local).interpret(
            "I don't have an additional preference for style.", state
        )
        self.assertIn("style", state.exhausted_attributes)
        self.assertIn("style", state.asked_attributes)


if __name__ == "__main__":
    unittest.main()
