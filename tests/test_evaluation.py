from __future__ import annotations

import dataclasses
import json
import unittest

from tikitaka.evaluation.ablations import compare_reports
from tikitaka.evaluation.experiment import ExperimentConfig, evaluate_samples
from tikitaka.evaluation.reporting import build_report, canonical_report_json, normalized_metrics_json
from tikitaka.evaluation.splits import SplitSpec, create_split, partition_samples


def config(**changes: object) -> ExperimentConfig:
    values = {
        "name": "test",
        "config_version": "1",
        "prompt_version": "prompt-1",
        "schema_version": "schema-1",
        "routing_mode": "pinned",
        "generative_provider": "none",
        "generative_model": "fake",
        "reasoning_level": "none",
        "retrieval_policy": "sparse",
        "embedding_route_id": "none",
        "index_id": "index-1",
        "reranker_route_id": "deterministic",
        "fusion_parameters": (("rrf_k", 60.0),),
        "profile_weight": 0.0,
        "question_policy": "never-ask",
        "seed": 7,
        "split_version": "split-v1",
        "catalog_checksum": "catalog-1",
        "code_revision": "abc123",
    }
    values.update(changes)
    return ExperimentConfig(**values)


def samples() -> list[dict]:
    result = []
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        for index in range(4):
            result.append({"sample_id": f"{scenario}-{index}", "scenario_type": scenario})
    return result


class RecordingAgent:
    def __init__(self) -> None:
        self.reset_calls: list[tuple] = []
        self.respond_calls: list[tuple] = []

    def reset(self, *args: object) -> None:
        self.reset_calls.append(args)

    def respond(self, *args: object) -> dict:
        self.respond_calls.append(args)
        return {
            "message": "options",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A"}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "calls": 1,
                "provider": "fake-provider",
                "model": "fake-model",
                "reasoning_level": "none",
                "component": "interpreter",
                "route": "fake-route",
            },
        }


class SplitTests(unittest.TestCase):
    def test_split_is_stable_disjoint_complete_and_stratified(self) -> None:
        spec = SplitSpec("split-v1", 7, 0.5)
        first = create_split(samples(), spec)
        second = create_split(reversed(samples()), spec)
        self.assertEqual(first, second)
        self.assertFalse(set(first.tuning_ids) & set(first.held_out_ids))
        self.assertEqual(set(first.tuning_ids) | set(first.held_out_ids), {row["sample_id"] for row in samples()})
        self.assertTrue(all((tuning, held) == (2, 2) for _, tuning, held in first.scenario_counts))
        tuning, held_out = partition_samples(samples(), first)
        self.assertEqual(len(tuning), 8)
        self.assertEqual(len(held_out), 8)


class ExperimentTests(unittest.TestCase):
    def test_cache_key_changes_at_every_required_version_boundary(self) -> None:
        base = config()
        base_key = base.cache_key("retrieval", {"query": "shoe"})
        for field, value in (
            ("prompt_version", "prompt-2"),
            ("generative_model", "fake-2"),
            ("embedding_route_id", "embed-2"),
            ("index_id", "index-2"),
            ("catalog_checksum", "catalog-2"),
            ("code_revision", "def456"),
        ):
            changed = dataclasses.replace(base, **{field: value})
            self.assertNotEqual(base_key, changed.cache_key("retrieval", {"query": "shoe"}), field)

    def test_agent_boundary_receives_only_official_inputs_and_usage_is_attributed(self) -> None:
        product = {
            "parent_asin": "A", "title": "Blue shoe", "features": ["cotton"],
            "details": {}, "description": [], "categories": ["Shoes"], "price": 10,
        }
        sample = {
            "sample_id": "s1", "scenario_type": "buying", "user_profile": {"summary": "safe"},
            "ground_truth": {"parent_asin": "A"},
        }
        agent = RecordingAgent()
        result = evaluate_samples(
            lambda: agent, [sample], {"A"}, {"A": ["Shoes"]}, {"A": product}, config(), "tuning"
        )
        self.assertEqual(len(agent.reset_calls[0]), 2)
        self.assertEqual(len(agent.respond_calls[0]), 4)
        self.assertEqual(agent.reset_calls[0][1], {"summary": "safe"})
        boundary_values = repr(agent.reset_calls + agent.respond_calls)
        self.assertNotIn("ground_truth", boundary_values)
        self.assertNotIn("scenario_type", boundary_values)
        self.assertEqual(result["metrics"]["hit_rate_at_10"], 1.0)
        self.assertEqual(result["usage"]["total_tokens"], 3)
        self.assertEqual(result["usage_by_component_route"][0]["component"], "interpreter")

    def test_deterministic_config_produces_byte_stable_normalized_metrics(self) -> None:
        product = {
            "parent_asin": "A", "title": "Blue shoe", "features": ["cotton"],
            "details": {}, "description": [], "categories": ["Shoes"], "price": 10,
        }
        sample = {
            "sample_id": "s1", "scenario_type": "buying", "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }
        experiment_config = config()
        results = [
            evaluate_samples(
                RecordingAgent, [sample], {"A"}, {"A": ["Shoes"]}, {"A": product},
                experiment_config, "tuning",
            )
            for _ in range(2)
        ]
        manifest = create_split(samples(), SplitSpec("split-v1", 7, 0.5))
        reports = [build_report(experiment_config, manifest, result, result) for result in results]
        self.assertEqual(normalized_metrics_json(reports[0]), normalized_metrics_json(reports[1]))


class ReportingTests(unittest.TestCase):
    def _result(self) -> dict:
        return {
            "metrics": {"sample_count": 1, "hit_rate_at_10": 1.0, "mrr": 1.0, "mttc": 1.0, "efficiency": 1.0, "technical_score": 1.0},
            "scenario_metrics": {"buying": {"sample_count": 1, "hit_rate_at_10": 1.0, "mrr": 1.0, "mttc": 1.0, "efficiency": 1.0, "technical_score": 1.0}},
            "questions": {"count": 0, "asked_attribute_distribution": {}},
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "api_key": "do-not-leak"},
            "usage_by_component_route": [],
            "sessions": [{"sample_id": "buying-0", "scenario_type": "buying", "hit": True}],
        }

    def test_report_has_required_identity_metrics_and_redacts_credentials(self) -> None:
        manifest = create_split(samples(), SplitSpec("split-v1", 7, 0.5))
        report = build_report(config(), manifest, self._result(), self._result())
        encoded = canonical_report_json(report)
        self.assertIn('"catalog_checksum": "catalog-1"', encoded)
        self.assertIn('"scenario_metrics"', encoded)
        self.assertNotIn("do-not-leak", encoded)
        self.assertEqual(json.loads(encoded)["results"]["held_out"]["usage"]["api_key"], "[REDACTED]")

    def test_ablation_rejects_undeclared_identity_change_and_reports_deltas(self) -> None:
        manifest = create_split(samples(), SplitSpec("split-v1", 7, 0.5))
        result = self._result()
        baseline = build_report(config(), manifest, result, result)
        changed_config = config(index_id="index-2")
        candidate = build_report(changed_config, manifest, result, result)
        with self.assertRaises(ValueError):
            compare_reports(baseline, candidate, [])
        comparison = compare_reports(baseline, candidate, ["index_id"])
        self.assertEqual(comparison["metric_deltas"]["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
