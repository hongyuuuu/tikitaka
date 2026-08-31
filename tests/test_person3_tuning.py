from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from tests.test_decision import question_candidates
from tests.test_ranking import FakeState
from tikitaka.decision import (
    CONTRACT_ORDER_SELECTION,
    DiagnosticsConfig,
    GeneralityConfig,
    HIGHEST_VALUE_SELECTION,
    PHASE5_ARM_VERSION,
    PHASE5_CLARIFICATION_THRESHOLDS,
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

    def test_invalid_question_selection_strategy_is_rejected_early(self) -> None:
        with self.assertRaises(ValueError):
            ResponsePolicyConfig(question_selection_strategy="unknown")


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

    def test_existing_phase4_fingerprint_is_preserved(self) -> None:
        self.assertEqual(
            self.baseline.fingerprint,
            "f8a1fd680366e8503f9d53969acb00a0f7b3bbebf73acc1bd0a8d84afca56d45",
        )

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

    def test_fixed_ask_baseline_changes_only_question_selection(self) -> None:
        fixed = self.arms["fixed-ask-baseline"]
        conservative = self.arms["conservative-questions-deterministic"]
        self.assertEqual(fixed.version, PHASE5_ARM_VERSION)
        self.assertEqual(
            fixed.response.question_selection_strategy,
            CONTRACT_ORDER_SELECTION,
        )
        policy = ResponsePolicy(config=fixed.response)
        self.assertEqual(
            policy.question_value.selection_strategy, CONTRACT_ORDER_SELECTION
        )
        self.assertEqual(
            replace(
                fixed.response,
                question_selection_strategy=HIGHEST_VALUE_SELECTION,
            ),
            conservative.response,
        )
        self.assertFalse(fixed.enable_llm_reranker)
        self.assertEqual(fixed.profile_weight, 0.0)

    def test_every_threshold_has_fixed_and_paired_llm_controls(self) -> None:
        for threshold in PHASE5_CLARIFICATION_THRESHOLDS:
            code = f"{round(threshold * 1000):03d}"
            deterministic_name = (
                "conservative-questions-deterministic"
                if threshold == 0.070
                else f"post-no-info-threshold-{code}-deterministic"
            )
            fixed_name = (
                "fixed-ask-baseline"
                if threshold == 0.070
                else f"fixed-ask-threshold-{code}"
            )
            llm_name = (
                "conservative-llm-anchored"
                if threshold == 0.070
                else f"post-no-info-threshold-{code}-llm-anchored"
            )
            deterministic = self.arms[deterministic_name]
            fixed = self.arms[fixed_name]
            llm = self.arms[llm_name]
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    deterministic.response.information_gain_threshold,
                    threshold,
                )
                self.assertEqual(llm.response, deterministic.response)
                self.assertEqual(llm.ranking, deterministic.ranking)
                self.assertTrue(llm.enable_llm_reranker)
                self.assertEqual(
                    llm.changed_report_variables_from(deterministic),
                    ("reranker_route_id",),
                )
                self.assertEqual(
                    replace(
                        fixed.response,
                        question_selection_strategy=HIGHEST_VALUE_SELECTION,
                    ),
                    deterministic.response,
                )
                self.assertEqual(
                    fixed.changed_report_variables_from(deterministic),
                    ("question_policy",),
                )
                self.assertEqual(deterministic.profile_weight, 0.0)
                self.assertEqual(fixed.profile_weight, 0.0)
                self.assertEqual(llm.profile_weight, 0.0)

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


class Phase6ReleaseFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.tuning = json.loads(
            (root / "reports" / "p5-threshold-070.json").read_text(
                encoding="utf-8"
            )
        )
        cls.held_out = json.loads(
            (root / "reports" / "p5-threshold-070-held-out.json").read_text(
                encoding="utf-8"
            )
        )
        cls.selection = json.loads(
            (root / "reports" / "p5-selection.json").read_text(encoding="utf-8")
        )

    def test_frozen_release_arm_matches_the_executable_person3_config(self) -> None:
        arm = phase4_arm("conservative-questions-deterministic")
        config = self.held_out["experiment"]["configuration"]
        ablation = config["ablation_parameters"]

        self.assertEqual(arm.response.information_gain_threshold, 0.07)
        self.assertEqual(arm.profile_weight, 0.0)
        self.assertFalse(arm.enable_llm_reranker)
        self.assertEqual(arm.question_policy_id, config["question_policy"])
        self.assertEqual(arm.reranker_route_id, config["reranker_route_id"])
        self.assertEqual(arm.fingerprint, ablation["decision_arm_fingerprint"])
        self.assertEqual(ablation["generative_policy"], "deterministic")

    def test_selection_points_to_the_one_time_held_out_confirmation(self) -> None:
        held_out_fingerprint = self.held_out["experiment"]["fingerprint"]
        metrics = self.held_out["results"]["held_out"]["metrics"]

        self.assertEqual(
            self.selection["selected_report"],
            "reports/p5-threshold-070-held-out.json",
        )
        self.assertEqual(
            self.selection["selected_fingerprint"], held_out_fingerprint
        )
        self.assertEqual(metrics["sample_count"], 60)
        self.assertEqual(metrics["hit_rate_at_10"], 0.933333)
        self.assertEqual(metrics["mrr"], 0.590245)
        self.assertEqual(metrics["mttc"], 5.0)

    def test_tuning_rejects_fixed_ask_and_historical_llm_controls(self) -> None:
        root = Path(__file__).resolve().parents[1] / "reports"
        fixed = json.loads(
            (root / "p5-fixed-ask-070.json").read_text(encoding="utf-8")
        )
        llm = json.loads(
            (root / "p5-llm-anchored-070.json").read_text(encoding="utf-8")
        )
        selected_metrics = self.tuning["results"]["tuning"]["metrics"]

        for control in (fixed, llm):
            control_metrics = control["results"]["tuning"]["metrics"]
            self.assertGreater(
                selected_metrics["hit_rate_at_10"],
                control_metrics["hit_rate_at_10"],
            )


if __name__ == "__main__":
    unittest.main()
