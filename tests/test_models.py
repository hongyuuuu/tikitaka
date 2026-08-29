"""Person 1 model-gateway tests. Offline, no provider, no network."""

from __future__ import annotations

import unittest

from tikitaka.contracts.domain import (
    Candidate,
    Constraint,
    ContractViolation,
    ProfileBias,
    SearchPlan,
    StateOperation,
    TurnDecision,
    Usage,
)
from tikitaka.models import usage as usage_module
from tikitaka.models.base import CredentialMissing, ModelRoute, ModelTimeout
from tikitaka.models.fake import (
    FaultyInterpreter,
    HeuristicInterpreter,
    ScriptedInterpreter,
    classify_constraint,
    detect_exhaustion,
)
from tikitaka.state.schema import (
    STRUCTURED_OUTPUT_SCHEMA,
    clamp_unit,
    normalize_budget,
    normalize_value,
    parse,
)
from tikitaka.state.session import new_session

ROUTE = ModelRoute(
    route_id="primary/gpt-5.6-terra",
    provider="main-api",
    model="gpt-5.6-terra",
    reasoning_level="xhigh",
)


class ContractRecordTests(unittest.TestCase):
    def test_constraint_rejects_out_of_range_values(self) -> None:
        for kwargs in (
            {"source_turn": 0},
            {"source_turn": 11},
            {"confidence": 1.5},
            {"confidence": -0.1},
            {"intent_version": 0},
            {"attribute": "vibe"},
            {"polarity": "maybe"},
            {"strength": "medium"},
            {"status": "haunted"},
        ):
            with self.subTest(**kwargs):
                base = {
                    "attribute": "color",
                    "value": "Red",
                    "normalized_value": "red",
                    "polarity": "include",
                    "strength": "soft",
                    "source_turn": 1,
                    "confidence": 0.5,
                    "intent_version": 1,
                }
                base.update(kwargs)
                with self.assertRaises(ContractViolation):
                    Constraint(**base)  # type: ignore[arg-type]

    def test_state_operation_field_rules(self) -> None:
        with self.assertRaises(ContractViolation):
            StateOperation(operation="add", attribute="color")  # no value
        with self.assertRaises(ContractViolation):
            StateOperation(operation="reset", scope="attribute")
        with self.assertRaises(ContractViolation):
            StateOperation(
                operation="remove", attribute="budget", new_value="under $10"
            )
        with self.assertRaises(ContractViolation):
            StateOperation(
                operation="exclude",
                attribute="material",
                new_value="leather",
                polarity="include",
                strength="hard",
                confidence=0.9,
            )

    def test_turn_decision_enforces_dg01(self) -> None:
        with self.assertRaises(ContractViolation):
            TurnDecision(
                action="clarify", ask_attribute=None, reason_code="final_turn"
            )
        with self.assertRaises(ContractViolation):
            TurnDecision(
                action="recommend",
                ask_attribute="color",
                reason_code="ranking_stable",
            )
        decision = TurnDecision(
            action="clarify",
            ask_attribute="material",
            reason_code="valuable_clarification",
            expected_information_gain=0.4,
        )
        self.assertEqual(decision.ask_attribute, "material")

    def test_dense_plan_requires_route_and_index(self) -> None:
        with self.assertRaises(ContractViolation):
            SearchPlan(route_policy="dense")
        plan = SearchPlan(
            route_policy="dense", embedding_route_id="e1", index_id="i1"
        )
        self.assertEqual(plan.index_id, "i1")

    def test_candidate_requires_valid_id_and_positive_ranks(self) -> None:
        with self.assertRaises(ContractViolation):
            Candidate(parent_asin="")
        with self.assertRaises(ContractViolation):
            Candidate(parent_asin="B01", sparse_rank=0)
        self.assertEqual(Candidate(parent_asin="B01").fused_score, 0.0)

    def test_profile_bias_defaults_to_inert(self) -> None:
        self.assertTrue(ProfileBias().is_inert)
        self.assertTrue(ProfileBias(terms=("fit",), weight=0.0).is_inert)
        self.assertFalse(ProfileBias(terms=("fit",), weight=0.2).is_inert)


class UsageTests(unittest.TestCase):
    def test_usage_cannot_go_negative(self) -> None:
        for kwargs in (
            {"prompt_tokens": -1},
            {"completion_tokens": -1},
            {"calls": -1},
            {"latency_ms": -1.0},
            {"estimated_cost": -0.01},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ContractViolation):
                    Usage(**kwargs)  # type: ignore[arg-type]

    def test_repairs_cannot_exceed_calls(self) -> None:
        with self.assertRaises(ContractViolation):
            Usage(calls=1, repairs=2)
        self.assertEqual(Usage(calls=2, repairs=1).repairs, 1)

    def test_merge_accumulates_and_keeps_left_identity(self) -> None:
        first = usage_module.for_route(
            ROUTE, prompt_tokens=100, completion_tokens=20, calls=1
        )
        repair = usage_module.for_route(
            ROUTE, prompt_tokens=120, completion_tokens=25, calls=1, repairs=1
        )
        merged = usage_module.merge(first, repair)
        self.assertEqual(merged.prompt_tokens, 220)
        self.assertEqual(merged.calls, 2)
        self.assertEqual(merged.repairs, 1)
        self.assertEqual(merged.model, "gpt-5.6-terra")
        self.assertEqual(merged.reasoning_level, "xhigh")

    def test_cache_hit_adds_no_call_tokens_or_cost(self) -> None:
        cached = usage_module.for_route(ROUTE, prompt_tokens=999, cache_hit=True)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.calls, 0)
        self.assertEqual(cached.prompt_tokens, 0)
        self.assertIsNone(cached.estimated_cost)

    def test_reasoning_tokens_are_priced_but_separately_reported(self) -> None:
        record = usage_module.for_route(
            ROUTE,
            prompt_tokens=1000,
            completion_tokens=1000,
            reasoning_tokens=1000,
            prompt_cost_per_1k=1.0,
            completion_cost_per_1k=2.0,
        )
        self.assertEqual(record.reasoning_tokens, 1000)
        self.assertEqual(record.estimated_cost, 5.0)
        self.assertEqual(record.total_tokens, 2000)

    def test_redacted_drops_provider_identity(self) -> None:
        record = usage_module.for_route(ROUTE, prompt_tokens=10)
        redacted = usage_module.redacted(record)
        self.assertIsNone(redacted.provider)
        self.assertIsNone(redacted.model)
        self.assertEqual(redacted.prompt_tokens, 10)


class ErrorTaxonomyTests(unittest.TestCase):
    def test_errors_carry_route_identity(self) -> None:
        error = ModelTimeout("no answer", ROUTE)
        self.assertIn("primary/gpt-5.6-terra", str(error))
        self.assertIs(error.route, ROUTE)

    def test_credential_error_does_not_carry_a_secret(self) -> None:
        error = CredentialMissing("credential environment variable is unset", ROUTE)
        rendered = f"{error!r} {error}"
        self.assertNotIn("sk-", rendered)
        self.assertIn("unset", rendered)


class NormalizationTests(unittest.TestCase):
    def test_clamp_unit_handles_hostile_input(self) -> None:
        self.assertEqual(clamp_unit(-1), 0.0)
        self.assertEqual(clamp_unit(0), 0.0)
        self.assertEqual(clamp_unit(0.5), 0.5)
        self.assertEqual(clamp_unit(1), 1.0)
        self.assertEqual(clamp_unit(2), 1.0)
        self.assertEqual(clamp_unit("high"), 0.0)
        self.assertEqual(clamp_unit(None), 0.0)
        self.assertEqual(clamp_unit(True), 0.0)
        self.assertEqual(clamp_unit(float("nan")), 0.0)

    def test_budget_parses_or_declines_to_invent(self) -> None:
        self.assertEqual(normalize_budget("under $80"), 80.0)
        self.assertEqual(normalize_budget("budget of 1,200"), 1200.0)
        self.assertEqual(normalize_budget(45), 45.0)
        self.assertIsNone(normalize_budget("cheap"))
        self.assertIsNone(normalize_budget(-5))

    def test_unparseable_budget_downgrades_to_other(self) -> None:
        self.assertEqual(normalize_value("budget", "as cheap as possible")[0], "other")
        self.assertEqual(normalize_value("budget", "under $30"), ("budget", 30.0))

    def test_structured_output_schema_is_closed(self) -> None:
        self.assertFalse(STRUCTURED_OUTPUT_SCHEMA["additionalProperties"])
        operations = STRUCTURED_OUTPUT_SCHEMA["properties"]["operations"]
        self.assertFalse(operations["items"]["additionalProperties"])
        self.assertIn("no_preference", operations["items"]["properties"]["operation"]["enum"])

    def test_operation_flood_is_truncated_deterministically(self) -> None:
        payload = {
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
        first = parse(payload)
        second = parse(payload)
        self.assertLessEqual(len(first.delta.operations), 24)
        self.assertEqual(first.delta.operations, second.delta.operations)
        self.assertTrue(any("flood" in error for error in first.errors))


class FakeInterpreterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = new_session("s", {})

    def test_heuristic_is_deterministic(self) -> None:
        interpreter = HeuristicInterpreter()
        message = "I'm looking for boots. A key requirement is: waterproof leather."
        first, _ = interpreter.interpret(message, self.state)
        second, _ = interpreter.interpret(message, self.state)
        self.assertEqual(first, second)

    def test_scripted_replays_by_turn_and_never_runs_out(self) -> None:
        interpreter = HeuristicInterpreter()
        one, _ = interpreter.interpret("I'm looking for hats.", self.state)
        two, _ = interpreter.interpret(
            "I'm looking for hats, but I'm still exploring.", self.state
        )
        scripted = ScriptedInterpreter([one, two])
        self.assertEqual(scripted.interpret("x", self.state)[0], one)
        self.assertEqual(scripted.interpret("x", self.state)[0], two)
        self.assertEqual(scripted.interpret("x", self.state)[0], two)

    def test_faulty_modes_behave_as_declared(self) -> None:
        with self.assertRaises(ModelTimeout):
            FaultyInterpreter(mode="timeout").interpret("x", self.state)
        with self.assertRaises(RuntimeError):
            FaultyInterpreter(mode="exception").interpret("x", self.state)
        delta, _ = FaultyInterpreter(mode="malformed").interpret("x", self.state)
        self.assertEqual(delta.operations, ())
        delta, _ = FaultyInterpreter(mode="unknown_operation").interpret("x", self.state)
        self.assertEqual(len(delta.operations), 1)
        self.assertEqual(delta.rejected_operations, 1)

    def test_classifier_matches_the_simulator_buckets(self) -> None:
        self.assertEqual(classify_constraint("budget under $50"), "budget")
        self.assertEqual(classify_constraint("full grain leather"), "material")
        self.assertEqual(classify_constraint("black finish"), "color")
        self.assertEqual(classify_constraint("wide width"), "size")
        self.assertEqual(classify_constraint("crew neck"), "style")
        self.assertEqual(classify_constraint("for hiking"), "use_case")
        self.assertEqual(classify_constraint("machine washable"), "feature")

    def test_exhaustion_detection_is_distinct_from_boundary(self) -> None:
        self.assertEqual(
            detect_exhaustion("I don't have an additional preference for size."), "size"
        )
        self.assertIsNone(
            detect_exhaustion(
                "I don't have a preference for size; please use your judgment."
            )
        )


if __name__ == "__main__":
    unittest.main()
