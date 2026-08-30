from __future__ import annotations

import unittest
from pathlib import Path

from tests.test_ranking import FakeConstraint, FakeState, candidate
from tikitaka.decision.diagnostics import DiagnosticsConfig, diagnose_pool
from tikitaka.decision.generality import GeneralityAssessment, GeneralitySensor
from tikitaka.decision.intent_router import VisibleModePolicy
from tikitaka.decision.phrasing import (
    LLMClarifier,
    LLMClarifierConfig,
    TextModelClarificationModel,
    clarification_message,
    recommendation_message,
)
from tikitaka.decision.question_value import (
    CONTRACT_ORDER_SELECTION,
    HIGHEST_VALUE_SELECTION,
    AttributeQuestionValue,
    QuestionValueEstimator,
    QuestionValueResult,
    _ordered_question_values,
    _ranking_change,
)
from tikitaka.decision.response_policy import ResponsePolicy, ResponsePolicyConfig
from tikitaka.contracts import DecisionPolicy, TurnDecision, Usage
from tikitaka.models.base import ModelRoute


def question_candidates(count: int = 14):
    result = []
    for index in range(count):
        material = "canvas" if index % 2 == 0 else "leather"
        result.append(
            candidate(
                f"P{index:02d}",
                1.0 - index * 0.035,
                values={
                    "category": ("shoes",),
                    "material": (material,),
                    "color": ("blue",),
                },
                sparse_rank=index + 1,
                dense_rank=count - index,
            )
        )
    return result


class DiagnosticsTests(unittest.TestCase):
    def test_diffuse_pool_has_more_effective_mass_than_concentrated_pool(self) -> None:
        state = FakeState()
        diffuse = [candidate(str(index), 0.5) for index in range(12)]
        concentrated = [candidate("LEAD", 1.0)] + [
            candidate(str(index), 0.0) for index in range(11)
        ]
        diffuse_result = diagnose_pool(state, diffuse)
        concentrated_result = diagnose_pool(state, concentrated)
        self.assertGreater(
            diffuse_result.effective_candidate_mass,
            concentrated_result.effective_candidate_mass,
        )
        self.assertGreater(concentrated_result.lead_margin, diffuse_result.lead_margin)

    def test_route_disagreement_uses_competitive_route_sets(self) -> None:
        candidates = [
            candidate("A", 1.0, sparse_rank=1, dense_rank=4),
            candidate("B", 0.9, sparse_rank=2, dense_rank=3),
            candidate("C", 0.8, sparse_rank=3, dense_rank=1),
            candidate("D", 0.7, sparse_rank=4, dense_rank=2),
        ]
        result = diagnose_pool(
            FakeState(), candidates, DiagnosticsConfig(route_top_k=2)
        )
        self.assertEqual(result.route_disagreement, 1.0)

    def test_missing_metadata_does_not_force_full_confidence(self) -> None:
        result = GeneralitySensor().assess(
            FakeState(), [candidate("A", 1.0), candidate("B", 0.9)]
        )
        self.assertLess(result.evidence_confidence, 0.5)

    def test_diagnostics_are_deterministic(self) -> None:
        candidates = question_candidates()
        self.assertEqual(
            diagnose_pool(FakeState(), candidates),
            diagnose_pool(FakeState(), candidates),
        )


class QuestionValueTests(unittest.TestCase):
    def test_rank_changing_material_is_selected(self) -> None:
        result = QuestionValueEstimator().estimate(
            FakeState(mode="browsing"), question_candidates(), turn=2
        )
        self.assertEqual(result.best_attribute, "material")
        self.assertGreater(result.expected_information_gain, 0.0)
        self.assertLessEqual(result.expected_information_gain, 1.0)

    def test_answered_attribute_is_removed(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        result = QuestionValueEstimator().estimate(state, question_candidates(), turn=2)
        self.assertNotIn("material", [item.attribute for item in result.values])

    def test_revalidation_attribute_can_be_asked_again(self) -> None:
        constraint = FakeConstraint(
            "material", "canvas", "canvas", status="needs_revalidation"
        )
        # It is deliberately present in both views to exercise the explicit
        # revalidation override in the structural contract.
        state = FakeState(
            active_constraints=(constraint,),
            revalidation_constraints=(constraint,),
            asked_attributes=frozenset({"material"}),
        )
        result = QuestionValueEstimator().estimate(state, question_candidates(), turn=2)
        self.assertIn("material", [item.attribute for item in result.values])

    def test_no_preference_and_asked_attributes_are_suppressed(self) -> None:
        state = FakeState(
            no_preference=frozenset({"material"}),
            asked_attributes=frozenset({"color"}),
        )
        result = QuestionValueEstimator().estimate(state, question_candidates(), turn=2)
        attributes = [item.attribute for item in result.values]
        self.assertNotIn("material", attributes)
        self.assertNotIn("color", attributes)

    def test_exhausted_attribute_is_suppressed(self) -> None:
        state = FakeState(exhausted_attributes=frozenset({"material"}))
        result = QuestionValueEstimator().estimate(
            state, question_candidates(), turn=2
        )
        self.assertNotIn("material", [item.attribute for item in result.values])

    def test_revalidation_does_not_reopen_exhausted_attribute(self) -> None:
        constraint = FakeConstraint(
            "material", "canvas", "canvas", status="needs_revalidation"
        )
        state = FakeState(
            active_constraints=(constraint,),
            revalidation_constraints=(constraint,),
            asked_attributes=frozenset({"material"}),
            exhausted_attributes=frozenset({"material"}),
        )
        result = QuestionValueEstimator().estimate(
            state, question_candidates(), turn=2
        )
        self.assertNotIn("material", [item.attribute for item in result.values])

    def test_fixed_selection_uses_contract_order_not_largest_gain(self) -> None:
        material = AttributeQuestionValue("material", 0.01, 1.0, 2, {})
        color = AttributeQuestionValue("color", 0.90, 1.0, 2, {})
        values = (color, material)

        adaptive = _ordered_question_values(values, HIGHEST_VALUE_SELECTION)
        fixed = _ordered_question_values(values, CONTRACT_ORDER_SELECTION)

        self.assertEqual(adaptive[0].attribute, "color")
        self.assertEqual(fixed[0].attribute, "material")

    def test_fixed_estimator_selects_first_eligible_real_attribute(self) -> None:
        candidates = [
            candidate(
                f"P{index:02d}",
                1.0 - index * 0.035,
                values={
                    "category": ("shoes",),
                    "material": ("canvas" if index < 7 else "leather",),
                    "color": ("blue" if index % 2 == 0 else "red",),
                },
                sparse_rank=index + 1,
                dense_rank=14 - index,
            )
            for index in range(14)
        ]
        adaptive = QuestionValueEstimator().estimate(
            FakeState(mode="browsing"), candidates, turn=2
        )
        fixed = QuestionValueEstimator(
            selection_strategy=CONTRACT_ORDER_SELECTION
        ).estimate(FakeState(mode="browsing"), candidates, turn=2)

        self.assertEqual(adaptive.best_attribute, "color")
        self.assertEqual(fixed.best_attribute, "material")
        self.assertLess(
            fixed.expected_information_gain,
            adaptive.expected_information_gain,
        )

    def test_unknown_question_selection_strategy_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            QuestionValueEstimator(selection_strategy="not-a-strategy")

    def test_turn_ten_has_no_question_value(self) -> None:
        result = QuestionValueEstimator().estimate(FakeState(), question_candidates(), 10)
        self.assertIsNone(result.best_attribute)
        self.assertEqual(result.expected_information_gain, 0.0)

    def test_sparse_attribute_metadata_is_ineligible(self) -> None:
        candidates = question_candidates()
        candidates[0] = candidate(
            "P00",
            1.0,
            values={"feature": ("waterproof",), "category": ("shoes",)},
        )
        result = QuestionValueEstimator().estimate(FakeState(), candidates, 2)
        self.assertNotIn("feature", [item.attribute for item in result.values])

    def test_answer_probabilities_follow_relevance_not_catalog_counts(self) -> None:
        candidates = [
            candidate(
                "LIKELY",
                1.0,
                values={"category": ("shoes",), "material": ("leather",)},
            )
        ] + [
            candidate(
                f"WEAK{index}",
                0.10,
                values={"category": ("shoes",), "material": ("canvas",)},
            )
            for index in range(6)
        ]
        result = QuestionValueEstimator().estimate(FakeState(), candidates, 2)
        material = next(item for item in result.values if item.attribute == "material")
        self.assertGreater(
            material.branch_probabilities["leather"],
            material.branch_probabilities["canvas"],
        )

    def test_rank_change_rewards_top_ten_and_reciprocal_rank_movement(self) -> None:
        base = tuple("ABCDEFGHIJKL")
        probabilities = {item: 1.0 / len(base) for item in base}
        crosses_boundary = tuple("ABCDEFGHIKJL")
        outside_only = tuple("ABCDEFGHIJLK")
        boundary_value = _ranking_change(
            base, crosses_boundary, 10, 0.75, 0.25, probabilities
        )
        outside_value = _ranking_change(
            base, outside_only, 10, 0.75, 0.25, probabilities
        )
        self.assertGreater(boundary_value, outside_value)


class StubGenerality:
    def __init__(self, score: float) -> None:
        self.score = score

    def assess(self, state, candidates):
        return GeneralityAssessment(self.score, 1.0, diagnose_pool(state, candidates))


class StubQuestionValue:
    def __init__(self, attribute: str | None, value: float) -> None:
        self.result = QuestionValueResult(attribute, value, ())

    def estimate(self, state, candidates, turn):
        return self.result


class BrokenQuestionValue:
    def estimate(self, state, candidates, turn):
        raise RuntimeError("broken")


class ResponsePolicyTests(unittest.TestCase):
    def test_default_policy_returns_frozen_contract_type(self) -> None:
        policy = ResponsePolicy(
            StubGenerality(0.1), StubQuestionValue("material", 0.8)
        )
        decision = policy.choose(FakeState(), question_candidates(), 2)
        self.assertIsInstance(policy, DecisionPolicy)
        self.assertIsInstance(decision, TurnDecision)

    def test_turn_ten_always_recommends(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(1.0), StubQuestionValue("material", 1.0)
        ).choose(FakeState(), question_candidates(), 10)
        self.assertEqual(decision.action, "recommend")
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason_code, "final_turn")

    def test_turn_nine_may_clarify_when_value_is_high(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.9),
            StubQuestionValue("material", 0.8),
            config=ResponsePolicyConfig(information_gain_threshold=0.05),
        ).choose(FakeState(mode="browsing"), question_candidates(), 9)
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.ask_attribute, "material")

    def test_low_generality_recommends(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.1), StubQuestionValue("material", 0.9)
        ).choose(FakeState(), question_candidates(), 2)
        self.assertEqual(decision.reason_code, "ranking_stable")

    def test_low_information_gain_recommends(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.9),
            StubQuestionValue("material", 0.01),
            config=ResponsePolicyConfig(information_gain_threshold=0.1),
        ).choose(FakeState(), question_candidates(), 2)
        self.assertEqual(decision.reason_code, "low_question_value")

    def test_turn_cost_suppresses_only_late_marginal_question(self) -> None:
        policy = ResponsePolicy(
            StubGenerality(0.9),
            StubQuestionValue("material", 0.08),
            config=ResponsePolicyConfig(
                information_gain_threshold=0.05,
                browsing_information_gain_adjustment=0.0,
            ),
        )
        early = policy.choose(FakeState(mode="browsing"), question_candidates(), 2)
        late = policy.choose(FakeState(mode="browsing"), question_candidates(), 9)
        self.assertEqual(early.action, "clarify")
        self.assertEqual(late.action, "recommend")
        self.assertEqual(late.reason_code, "low_question_value")

    def test_no_eligible_attribute_recommends(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.9), StubQuestionValue(None, 0.0)
        ).choose(FakeState(), question_candidates(), 2)
        self.assertEqual(decision.reason_code, "no_eligible_attribute")

    def test_empty_pool_recommends_with_insufficient_evidence(self) -> None:
        decision = ResponsePolicy().choose(FakeState(), [], 2)
        self.assertEqual(decision.reason_code, "insufficient_evidence")

    def test_component_failure_uses_recommend_fallback(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.9), BrokenQuestionValue()
        ).choose(FakeState(), question_candidates(), 2)
        self.assertEqual(decision.action, "recommend")
        self.assertEqual(decision.reason_code, "component_fallback")

    def test_mutual_exclusion_invariant(self) -> None:
        clarify = ResponsePolicy(
            StubGenerality(0.9), StubQuestionValue("material", 0.8)
        ).choose(FakeState(mode="browsing"), question_candidates(), 2)
        recommend = ResponsePolicy(
            StubGenerality(0.1), StubQuestionValue("material", 0.8)
        ).choose(FakeState(), question_candidates(), 2)
        self.assertIsNotNone(clarify.ask_attribute)
        self.assertIsNone(recommend.ask_attribute)

    def test_fixed_order_reason_does_not_claim_maximum_information_gain(self) -> None:
        decision = ResponsePolicy(
            StubGenerality(0.9),
            StubQuestionValue("material", 0.8),
            config=ResponsePolicyConfig(
                question_selection_strategy=CONTRACT_ORDER_SELECTION
            ),
        ).choose(FakeState(mode="browsing"), question_candidates(), 2)
        self.assertEqual(decision.action, "clarify")
        self.assertIn("pinned contract order", decision.reason)
        self.assertNotIn("greatest expected", decision.reason)


class ModeAndPhrasingTests(unittest.TestCase):
    def test_trusted_visible_mode_is_used(self) -> None:
        self.assertEqual(VisibleModePolicy().resolve(FakeState(mode="browsing")), "browsing")

    def test_unknown_mode_fallback_uses_visible_hard_constraints(self) -> None:
        state = FakeState(
            mode="unknown",
            mode_confidence=0.0,
            active_constraints=(
                FakeConstraint("category", "shoes", "shoes"),
                FakeConstraint("budget", 80, 80),
            ),
        )
        self.assertEqual(VisibleModePolicy().resolve(state), "buying")

    def test_clarification_phrase_uses_candidate_supported_examples(self) -> None:
        message = clarification_message("material", question_candidates(4))
        self.assertIn("canvas", message)
        self.assertIn("leather", message)

    def test_clarification_examples_prefer_relevant_supported_branches(self) -> None:
        candidates = [
            candidate("OUTLIER", 1.0, values={"material": ("cork",)}),
            *[
                candidate(
                    f"CANVAS{index}",
                    0.9 - index * 0.01,
                    values={"material": ("canvas",)},
                )
                for index in range(4)
            ],
            *[
                candidate(
                    f"LEATHER{index}",
                    0.8 - index * 0.01,
                    values={"material": ("leather",)},
                )
                for index in range(4)
            ],
        ]
        message = clarification_message("material", candidates)
        self.assertIn("canvas", message)
        self.assertIn("leather", message)
        self.assertNotIn("cork", message)

    def test_other_phrase_does_not_bundle_attributes(self) -> None:
        message = clarification_message("other", question_candidates())
        self.assertEqual(message.count("?"), 1)

    def test_recommendation_message_handles_empty_result(self) -> None:
        self.assertIn("could not find", recommendation_message(0))

    def test_llm_clarifier_cannot_change_structured_attribute(self) -> None:
        class Model:
            def clarify(self, request):
                self.request = request
                return "Would canvas or leather work better for you?", type(
                    "Usage", (), {"calls": 1, "prompt_tokens": 5, "completion_tokens": 3}
                )()

        model = Model()
        message, usage = LLMClarifier(model).phrase("material", question_candidates(4))
        self.assertIn("canvas", message)
        self.assertEqual(model.request.ask_attribute, "material")
        self.assertEqual(usage.route, "llm_clarifier")

    def test_malformed_llm_phrase_uses_deterministic_fallback(self) -> None:
        class Model:
            def clarify(self, request):
                return "What material? What color?", type("Usage", (), {"calls": 1})()

        message, usage = LLMClarifier(Model()).phrase("material", question_candidates(4))
        self.assertEqual(message.count("?"), 1)
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.route, "clarification_fallback")

    def test_single_question_about_wrong_attribute_uses_fallback(self) -> None:
        class Model:
            def clarify(self, request):
                return "Which color would you like?", Usage(calls=1)

        message, usage = LLMClarifier(Model()).phrase(
            "material", question_candidates(4)
        )
        self.assertIn("material", message.lower())
        self.assertNotIn("color", message.lower())
        self.assertEqual(usage.route, "clarification_fallback")

    def test_example_substring_cannot_fake_question_grounding(self) -> None:
        class Model:
            def clarify(self, request):
                return "Which brand do you prefer?", Usage(calls=1)

        candidates = [
            candidate("A", 1.0, values={"color": ("red",)}),
            candidate("B", 0.9, values={"color": ("blue",)}),
        ]
        message, usage = LLMClarifier(Model()).phrase("color", candidates)
        self.assertIn("color", message.lower())
        self.assertNotIn("brand", message.lower())
        self.assertEqual(usage.route, "clarification_fallback")

    def test_provider_neutral_clarifier_uses_strict_grounded_schema(self) -> None:
        class TextModel:
            def __init__(self):
                self.calls = []

            def complete_structured(self, prompt, schema, route):
                self.calls.append((prompt, schema, route))
                return {"question": "Would canvas or leather suit you better?"}, Usage(
                    prompt_tokens=9,
                    completion_tokens=4,
                    calls=1,
                    provider="fake",
                    model=route.model,
                    reasoning_level=route.reasoning_level,
                    route=route.route_id,
                )

        route = ModelRoute(
            route_id="primary/gpt-5.6-terra",
            provider="fake",
            model="gpt-5.6-terra",
            reasoning_level="medium",
        )
        text_model = TextModel()
        adapter = TextModelClarificationModel(text_model, route)
        message, usage = LLMClarifier(adapter).phrase(
            "material", question_candidates(4)
        )
        prompt, schema, used_route = text_model.calls[0]
        self.assertIn('"ask_attribute":"material"', prompt)
        self.assertIn('"supported_examples":["canvas","leather"]', prompt)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(used_route, route)
        self.assertIn("canvas", message)
        self.assertEqual(usage.prompt_tokens, 9)

    def test_clarifier_failure_preserves_billed_usage(self) -> None:
        class BilledFailure(RuntimeError):
            def __init__(self):
                super().__init__("malformed")
                self.usage = Usage(
                    prompt_tokens=7,
                    completion_tokens=1,
                    calls=1,
                    provider="fake",
                )

        class Model:
            def clarify(self, request):
                raise BilledFailure()

        _, usage = LLMClarifier(Model()).phrase("material", question_candidates(4))
        self.assertEqual(usage.prompt_tokens, 7)
        self.assertEqual(usage.route, "clarification_fallback")

    def test_llm_clarifier_exception_counts_attempt_and_falls_back(self) -> None:
        class Model:
            def clarify(self, request):
                raise TimeoutError()

        message, usage = LLMClarifier(Model()).phrase("material", question_candidates(4))
        self.assertIn("material", message.lower())
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.route, "clarification_fallback")

    def test_llm_clarifier_rejects_unapproved_model(self) -> None:
        with self.assertRaises(ValueError):
            LLMClarifierConfig(model="another-model")


class RuntimeBoundaryTests(unittest.TestCase):
    def test_person3_runtime_has_no_hidden_label_access(self) -> None:
        package_root = Path(__file__).parents[1] / "tikitaka"
        forbidden = ("ground_truth", "scenario_type", "public_set", "intent_card")
        for directory in (package_root / "decision", package_root / "ranking"):
            for path in directory.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, source, f"{token} leaked into {path}")

    def test_person3_runtime_has_no_provider_sdk_import(self) -> None:
        package_root = Path(__file__).parents[1] / "tikitaka"
        for directory in (package_root / "decision", package_root / "ranking"):
            for path in directory.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("import openai", source.lower())
                self.assertNotIn("from openai", source.lower())


if __name__ == "__main__":
    unittest.main()
