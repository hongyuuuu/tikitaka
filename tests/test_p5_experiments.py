from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from tikitaka.evaluation import P5ExperimentArm, select_release_report
from tikitaka.evaluation.experiment import ExperimentConfig
from tikitaka.orchestration.runtime import build_deterministic_agent


CATALOG = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


def _report(hit: float, mrr: float, mttc: float, *, buying: float | None = None) -> dict:
    scenario_hit = hit if buying is None else buying
    return {
        "results": {
            "held_out": {
                "metrics": {"hit_rate_at_10": hit, "mrr": mrr, "mttc": mttc},
                "scenario_metrics": {
                    name: {"hit_rate_at_10": scenario_hit if name == "buying" else hit}
                    for name in ("buying", "browsing", "intent_override", "boundary")
                },
            }
        }
    }


class P5ArmTests(unittest.TestCase):
    def test_arm_identity_covers_every_executable_axis(self) -> None:
        baseline = P5ExperimentArm("baseline")
        alternatives = (
            dataclasses.replace(baseline, retrieval_policy="hybrid"),
            dataclasses.replace(baseline, generative_policy="api_selective"),
            dataclasses.replace(baseline, decision_arm="always-recommend-baseline"),
            dataclasses.replace(baseline, profile_weight=0.1),
        )
        self.assertTrue(all(item.fingerprint != baseline.fingerprint for item in alternatives))
        self.assertEqual(baseline.selected_profile_weight, 0.0)

    def test_unknown_or_out_of_range_arm_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            P5ExperimentArm("bad", retrieval_policy="magic")
        with self.assertRaises(ValueError):
            P5ExperimentArm("bad", generative_policy="sometimes")
        with self.assertRaises(KeyError):
            P5ExperimentArm("bad", decision_arm="missing")
        with self.assertRaises(ValueError):
            P5ExperimentArm("bad", profile_weight=1.1)

    def test_report_parameters_are_stable_and_credential_free(self) -> None:
        arm = P5ExperimentArm("hybrid", retrieval_policy="hybrid")
        parameters = dict(arm.report_parameters())
        self.assertEqual(parameters["generative_policy"], "deterministic")
        self.assertNotIn("key", repr(parameters).casefold())


class P5SelectionTests(unittest.TestCase):
    def test_official_objective_order_is_lexicographic(self) -> None:
        baseline = _report(0.80, 0.50, 6.0)
        better_mrr = _report(0.80, 0.55, 7.0)
        faster_but_lower_hit = _report(0.79, 0.90, 1.0)
        self.assertIs(
            select_release_report((baseline, faster_but_lower_hit, better_mrr)),
            better_mrr,
        )

    def test_material_scenario_collapse_rejects_aggregate_winner(self) -> None:
        baseline = _report(0.80, 0.50, 6.0)
        collapsed = _report(0.90, 0.70, 4.0, buying=0.70)
        self.assertIs(select_release_report((baseline, collapsed)), baseline)


class P5RuntimeSeamTests(unittest.TestCase):
    def test_person4_can_inject_a_pinned_retrieval_route(self) -> None:
        class EmptyRetriever:
            def search(self, plan: object, limit: int) -> list:
                return []

        retriever = EmptyRetriever()
        agent = build_deterministic_agent(CATALOG, retriever=retriever)
        self.assertIs(agent._retriever, retriever)


class P5ExperimentIdentityTests(unittest.TestCase):
    def test_ablation_parameters_change_experiment_fingerprint(self) -> None:
        values = dict(
            name="p5",
            config_version="1",
            prompt_version="p",
            schema_version="s",
            routing_mode="pinned",
            generative_provider="none",
            generative_model="heuristic",
            reasoning_level="none",
            retrieval_policy="sparse",
            embedding_route_id="none",
            index_id="catalog:x",
            reranker_route_id="deterministic",
            fusion_parameters=(),
            profile_weight=0.0,
            question_policy="adaptive",
            seed=1,
            split_version="v1",
            catalog_checksum="x",
            code_revision="abc",
        )
        baseline = ExperimentConfig(**values)
        changed = ExperimentConfig(
            **values, ablation_parameters=(("generative_policy", "api_selective"),)
        )
        self.assertNotEqual(baseline.fingerprint, changed.fingerprint)
        self.assertEqual(
            changed.to_dict()["ablation_parameters"]["generative_policy"],
            "api_selective",
        )


if __name__ == "__main__":
    unittest.main()
