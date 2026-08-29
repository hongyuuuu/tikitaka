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
    clarification_message,
    recommendation_message,
)
from tikitaka.decision.question_value import QuestionValueEstimator, QuestionValueResult
from tikitaka.decision.response_policy import ResponsePolicy, ResponsePolicyConfig
from tikitaka.contracts import DecisionPolicy, TurnDecision


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
            active_constraints=(constraint,), revalidation_constraints=(constraint,)
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
