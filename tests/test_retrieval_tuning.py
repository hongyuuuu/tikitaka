from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.retrieval_fakes import (
    create_failing_gateway_semantic_embedder,
    create_gateway_semantic_embedder,
)
from scripts.sweep_sparse_runtime import _build_agent, _experiment_config
from tikitaka.retrieval.benchmark import load_retrieval_benchmark_cases
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact, load_dense_index
from tikitaka.retrieval.manifests import dense_manifest_as_dict
from tikitaka.retrieval.sparse import SparseIndexConfig
from tikitaka.retrieval.tuning import (
    SweepExecutionError,
    SweepValidationError,
    build_retrieval_sweep_report,
    load_retrieval_sweep_spec,
    run_retrieval_variant,
    select_retrieval_variant,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "fixtures" / "retrieval_benchmark_catalog.jsonl"
CASES = ROOT / "tests" / "fixtures" / "retrieval_benchmark_cases.jsonl"
SPEC = ROOT / "configs" / "retrieval_m4_sweep.json"


class RetrievalSweepSpecTests(unittest.TestCase):
    def test_default_spec_covers_the_required_m4_ablation_surfaces(self) -> None:
        spec = load_retrieval_sweep_spec(SPEC)

        by_id = {variant.variant_id: variant for variant in spec.variants}
        self.assertEqual(spec.baseline_variant_id, "sparse-baseline")
        self.assertEqual(spec.selection_k, 10)
        self.assertEqual(
            {variant.route for variant in spec.variants},
            {"sparse", "dense", "hybrid", "auto"},
        )
        self.assertFalse(by_id["sparse-boost-only"].ranking.hard_filtering)
        self.assertEqual(by_id["sparse-profile-soft"].request_profile_weight, 0.25)
        self.assertEqual(by_id["hybrid-coverage"].hybrid.fused_depth, 300)
        self.assertEqual(by_id["hybrid-coverage"].candidate_limit, 200)
        self.assertEqual(spec.fingerprint, load_retrieval_sweep_spec(SPEC).fingerprint)

    def test_rejects_unknown_fields_nonfinite_values_and_duplicate_variants(self) -> None:
        payload = json.loads(SPEC.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            unknown = deepcopy(payload)
            unknown["variants"][0]["mystery"] = 1
            path.write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(SweepValidationError, "unknown variant fields"):
                load_retrieval_sweep_spec(path)

            duplicated = deepcopy(payload)
            duplicated["variants"][1]["variant_id"] = "sparse-baseline"
            path.write_text(json.dumps(duplicated), encoding="utf-8")
            with self.assertRaisesRegex(SweepValidationError, "must be unique"):
                load_retrieval_sweep_spec(path)

            path.write_text(
                SPEC.read_text(encoding="utf-8").replace("8.0", "NaN", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SweepValidationError, "non-finite"):
                load_retrieval_sweep_spec(path)

        with self.assertRaisesRegex(ValueError, "finite"):
            SparseIndexConfig(title_weight=math.nan)


class RetrievalSweepExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(CATALOG)
        cls.cases = load_retrieval_benchmark_cases(
            CASES,
            valid_ids=cls.catalog.ids,
        )
        cls.spec = load_retrieval_sweep_spec(SPEC)
        cls.directory = tempfile.TemporaryDirectory()
        cls.embedder = create_gateway_semantic_embedder()
        cls.manifest = build_dense_artifact(
            cls.catalog,
            cls.embedder,
            cls.directory.name,
            embedding_provider=cls.embedder.provider,
            embedding_model=cls.embedder.model,
            batch_size=3,
        )
        cls.index = load_dense_index(cls.directory.name, cls.catalog)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_full_fixture_sweep_runs_every_route_without_selecting_a_winner(self) -> None:
        reports = [
            run_retrieval_variant(
                variant,
                self.cases,
                self.catalog,
                ks=(1, 3, 5, 10),
                dense_index=self.index,
                query_embedder=self.embedder,
            )
            for variant in self.spec.variants
        ]

        report = build_retrieval_sweep_report(
            self.spec,
            reports,
            evidence_tier="fixture",
            code_revision="fixture-revision",
            code_dirty=True,
            case_file=str(CASES),
            case_file_sha256="fixture-case-checksum",
            catalog=self.catalog,
            dense_manifest=dense_manifest_as_dict(self.manifest),
            dense_backend=self.index.backend,
        )

        self.assertEqual(len(report["variants"]), len(self.spec.variants))
        self.assertTrue(report["code_dirty"])
        self.assertIn(report["dense_backend"], {"numpy-exact", "python-exact"})
        self.assertEqual(report["selection"]["status"], "fixture_mechanics_only")
        self.assertIsNone(report["selection"]["selected_variant_id"])
        self.assertTrue(report["selection"]["provisional_tuning_leader"])
        self.assertFalse(
            report["selection"]["heldout_confirmation"]["used_for_selection"]
        )
        auto = next(item for item in reports if item["variant_id"] == "auto-runtime")
        self.assertEqual(auto["execution"]["executed_routes"], {"hybrid": 8})
        self.assertEqual(auto["execution"]["degraded_case_count"], 0)
        self.assertIn(
            "intent_override",
            auto["retrieval_diagnostics"]["splits"]["heldout"]["scenarios"],
        )

    def test_public_evidence_selects_from_tuning_and_only_confirms_on_heldout(self) -> None:
        variants = self.spec.variants[:2]
        reports = [
            run_retrieval_variant(
                variant,
                self.cases,
                self.catalog,
                ks=(10,),
            )
            for variant in variants
        ]
        reduced_spec = type(self.spec)(
            baseline_variant_id=variants[0].variant_id,
            selection_k=10,
            max_scenario_hit_rate_drop=0.0,
            variants=variants,
        )

        selection = select_retrieval_variant(
            reduced_spec,
            reports,
            evidence_tier="public-development",
        )

        self.assertEqual(selection["status"], "selected")
        self.assertIn(
            selection["selected_variant_id"],
            {variant.variant_id for variant in variants},
        )
        self.assertEqual(selection["selection_basis"], "tuning_only")
        self.assertFalse(selection["heldout_confirmation"]["used_for_selection"])

    def test_scenario_collapse_rejects_an_aggregate_tuning_leader(self) -> None:
        variants = self.spec.variants[:2]
        reports = [
            run_retrieval_variant(
                variant,
                self.cases,
                self.catalog,
                ks=(10,),
            )
            for variant in variants
        ]
        baseline, candidate = deepcopy(reports[0]), deepcopy(reports[1])
        baseline_tuning = baseline["metrics"]["splits"]["tuning"]
        candidate_tuning = candidate["metrics"]["splits"]["tuning"]
        baseline_tuning["overall"]["hit_rate_at_k"]["10"] = 0.5
        baseline_tuning["overall"]["mrr_at_k"]["10"] = 0.5
        candidate_tuning["overall"]["hit_rate_at_k"]["10"] = 0.9
        candidate_tuning["overall"]["mrr_at_k"]["10"] = 0.9
        baseline_tuning["scenarios"]["buying"]["hit_rate_at_k"]["10"] = 1.0
        candidate_tuning["scenarios"]["buying"]["hit_rate_at_k"]["10"] = 0.0
        reduced_spec = type(self.spec)(
            baseline_variant_id=variants[0].variant_id,
            selection_k=10,
            max_scenario_hit_rate_drop=0.0,
            variants=variants,
        )

        selection = select_retrieval_variant(
            reduced_spec,
            (baseline, candidate),
            evidence_tier="public-development",
        )

        self.assertEqual(selection["selected_variant_id"], variants[0].variant_id)
        guard = selection["scenario_collapse_guards"][variants[1].variant_id]
        self.assertFalse(guard["eligible"])
        self.assertEqual(guard["collapsed_scenarios"], ["buying"])

    def test_public_evidence_rejects_a_dirty_uncommitted_worktree(self) -> None:
        variants = self.spec.variants[:2]
        reports = [
            run_retrieval_variant(
                variant,
                self.cases,
                self.catalog,
                ks=(10,),
            )
            for variant in variants
        ]
        reduced_spec = type(self.spec)(
            baseline_variant_id=variants[0].variant_id,
            selection_k=10,
            max_scenario_hit_rate_drop=0.0,
            variants=variants,
        )
        with self.assertRaisesRegex(SweepValidationError, "clean committed"):
            build_retrieval_sweep_report(
                reduced_spec,
                reports,
                evidence_tier="public-development",
                code_revision="dirty-revision",
                code_dirty=True,
                case_file=str(CASES),
                case_file_sha256="fixture-case-checksum",
                catalog=self.catalog,
                dense_manifest=None,
                dense_backend=None,
            )

    def test_dense_failure_cannot_masquerade_as_a_quality_variant(self) -> None:
        dense = next(
            variant for variant in self.spec.variants if variant.variant_id == "dense-only"
        )
        with self.assertRaisesRegex(SweepExecutionError, "degraded"):
            run_retrieval_variant(
                dense,
                self.cases,
                self.catalog,
                ks=(10,),
                dense_index=self.index,
                query_embedder=create_failing_gateway_semantic_embedder(),
            )

    def test_sparse_runtime_experiment_applies_the_same_versioned_variant(self) -> None:
        baseline = self.spec.variants[0]
        revised = next(
            variant
            for variant in self.spec.variants
            if variant.variant_id == "sparse-title-category-heavy"
        )
        baseline_config = _experiment_config(
            baseline,
            self.spec,
            split_version="fixture-split",
            seed=7,
            catalog_checksum=self.catalog.identity.source_sha256,
            code_revision="fixture-revision",
        )
        revised_config = _experiment_config(
            revised,
            self.spec,
            split_version="fixture-split",
            seed=7,
            catalog_checksum=self.catalog.identity.source_sha256,
            code_revision="fixture-revision",
        )
        self.assertNotEqual(baseline_config.fingerprint, revised_config.fingerprint)
        self.assertIn(
            ("sparse.title_weight", 8.0),
            revised_config.fusion_parameters,
        )

        agent = _build_agent(self.catalog, revised)
        try:
            agent.reset("runtime-m4", {})
            response = agent.respond(
                "runtime-m4",
                "I need waterproof hiking shoes under $80.",
                1,
                10,
            )
            self.assertIsInstance(response["message"], str)
            self.assertLessEqual(len(response["recommendations"]), 10)
            self.assertEqual(agent._candidate_limit, revised.candidate_limit)
            self.assertEqual(agent._retriever.sparse.config.title_weight, 8.0)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
