from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_decision import question_candidates
from tests.test_ranking import FakeState
from tikitaka.decision import (
    DiagnosticsConfig,
    GeneralityConfig,
    QuestionValueConfig,
    ResponsePolicy,
    ResponsePolicyConfig,
    phase4_arm,
    phase4_experiment_arms,
)
from tikitaka.orchestration.runtime import RuntimeConfig
from tikitaka.ranking import DeterministicRankerConfig


class ResponsePolicyTuningTests(unittest.TestCase):
    def test_nested_person3_configs_reach_live_components(self) -> None:
        generality = GeneralityConfig(effective_mass_weight=0.40)
        diagnostics = DiagnosticsConfig(score_temperature=0.31)
        question = QuestionValueConfig(
            minimum_attribute_coverage=0.55,
            membership_weight=0.625,
            order_weight=0.375,
        )
        question_ranker = DeterministicRankerConfig(
            fused_weight=0.40,
            constraint_match_weight=0.30,
        )
        config = ResponsePolicyConfig(
            generality=generality,
            diagnostics=diagnostics,
            question_value=question,
            question_ranker=question_ranker,
        )

        policy = ResponsePolicy(config=config)

        self.assertEqual(policy.generality_sensor.config, generality)
        self.assertEqual(policy.generality_sensor.diagnostics_config, diagnostics)
        self.assertEqual(policy.question_value.config, question)
        self.assertEqual(policy.question_value.ranker.config, question_ranker)

    def test_pinned_always_recommend_arm_never_spends_a_question_turn(self) -> None:
        arm = phase4_arm("always-recommend-baseline")
        decision = ResponsePolicy(config=arm.response).choose(
            FakeState(mode="browsing"), question_candidates(), 1
        )
        self.assertEqual(decision.action, "recommend")
        self.assertIsNone(decision.ask_attribute)

    def test_invalid_nested_config_is_rejected_early(self) -> None:
        with self.assertRaises(TypeError):
            ResponsePolicyConfig(question_value=object())  # type: ignore[arg-type]


class Phase4ExperimentArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arms = {arm.name: arm for arm in phase4_experiment_arms()}
        self.baseline = self.arms["adaptive-deterministic"]

    def test_arms_have_unique_reproducible_fingerprints(self) -> None:
        first = phase4_experiment_arms()
        second = phase4_experiment_arms()
        self.assertEqual(
            [arm.fingerprint for arm in first],
            [arm.fingerprint for arm in second],
        )
        self.assertEqual(len({arm.fingerprint for arm in first}), len(first))
        self.assertTrue(all(len(arm.fingerprint) == 64 for arm in first))

    def test_official_proxy_normalizes_hit_and_mrr_weights(self) -> None:
        arm = self.arms["official-proxy-deterministic"]
        question = arm.response.question_value
        self.assertEqual(question.membership_weight, 0.625)
        self.assertEqual(question.order_weight, 0.375)

    def test_one_factor_arms_declare_only_the_changed_report_field(self) -> None:
        expected = {
            "official-proxy-deterministic": ("question_policy",),
            "adaptive-llm-unanchored": ("reranker_route_id",),
            "adaptive-llm-anchored": ("reranker_route_id",),
            "adaptive-profile-010": ("profile_weight",),
        }
        for name, fields in expected.items():
            with self.subTest(arm=name):
                self.assertEqual(
                    self.arms[name].changed_report_variables_from(self.baseline),
                    fields,
                )

    def test_runtime_accepts_arm_without_cross_owner_adapter_code(self) -> None:
        arm = self.arms["adaptive-llm-anchored"]
        runtime = RuntimeConfig(**dict(arm.runtime_overrides))
        self.assertEqual(runtime.decision, arm.response)
        self.assertEqual(runtime.ranking, arm.ranking)
        self.assertEqual(runtime.llm_reranker, arm.llm_reranker)
        self.assertTrue(runtime.enable_llm_reranker)

    def test_profile_arm_does_not_double_apply_ranker_profile_weight(self) -> None:
        arm = self.arms["adaptive-profile-010"]
        self.assertEqual(arm.profile_weight, 0.10)
        self.assertEqual(arm.ranking.profile_weight, 0.0)
        self.assertEqual(arm.response.question_ranker.profile_weight, 0.0)

    def test_fingerprint_changes_when_any_person3_parameter_changes(self) -> None:
        changed_response = replace(
            self.baseline.response,
            information_gain_threshold=0.056,
        )
        changed = replace(self.baseline, response=changed_response)
        self.assertNotEqual(changed.fingerprint, self.baseline.fingerprint)

    def test_unknown_arm_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            phase4_arm("not-an-arm")


if __name__ == "__main__":
    unittest.main()
