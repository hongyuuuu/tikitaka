from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tikitaka.retrieval.benchmark import (
    BenchmarkValidationError,
    evaluate_retrieval_route,
    load_retrieval_benchmark_cases,
)
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.retriever import SparseStructuredRetriever
from scripts.benchmark_retrieval import _diagnostic_report


CATALOG = Path(__file__).parent / "fixtures" / "retrieval_benchmark_catalog.jsonl"
CASES = Path(__file__).parent / "fixtures" / "retrieval_benchmark_cases.jsonl"


class RetrievalBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(CATALOG)

    def test_loads_explicit_tuning_and_heldout_cases(self) -> None:
        cases = load_retrieval_benchmark_cases(CASES, valid_ids=self.catalog.ids)

        self.assertEqual(len(cases), 8)
        self.assertEqual({case.split for case in cases}, {"tuning", "heldout"})
        self.assertEqual(cases[2].request.intent_version, 2)
        self.assertEqual(cases[3].request.no_preference, frozenset({"color"}))

    def test_reports_split_and_scenario_metrics_without_target_input(self) -> None:
        cases = load_retrieval_benchmark_cases(CASES, valid_ids=self.catalog.ids)
        seen_requests = []
        with SparseStructuredRetriever(self.catalog) as retriever:
            def search(request, limit):
                seen_requests.append(request)
                return retriever.search(request, limit)

            report = evaluate_retrieval_route(
                cases,
                search,
                valid_ids=self.catalog.ids,
                ks=(1, 3, 5),
            )

        self.assertEqual(len(seen_requests), len(cases))
        self.assertTrue(all(not hasattr(request, "target_parent_asin") for request in seen_requests))
        self.assertEqual(set(report["splits"]), {"tuning", "heldout"})
        self.assertEqual(report["splits"]["heldout"]["overall"]["case_count"], 4)
        self.assertEqual(
            set(report["splits"]["heldout"]["scenarios"]),
            {"buying", "browsing", "intent_override", "boundary"},
        )
        self.assertIn("intent_override", report["splits"]["heldout"]["scenarios"])

    def test_rejects_missing_split_invalid_target_duplicate_and_invalid_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(CASES.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkValidationError, "tuning and heldout"):
                load_retrieval_benchmark_cases(path, valid_ids=self.catalog.ids)

        cases = load_retrieval_benchmark_cases(CASES, valid_ids=self.catalog.ids)

        class Invalid:
            parent_asin = "NOT_IN_CATALOG"

        with self.assertRaisesRegex(BenchmarkValidationError, "outside the frozen catalog"):
            evaluate_retrieval_route(
                cases[:1],
                lambda request, limit: [Invalid()],
                valid_ids=self.catalog.ids,
                ks=(10,),
            )

        class Duplicate:
            parent_asin = "A_HIKE"

        with self.assertRaisesRegex(BenchmarkValidationError, "duplicate"):
            evaluate_retrieval_route(
                cases[:1],
                lambda request, limit: [Duplicate(), Duplicate()],
                valid_ids=self.catalog.ids,
                ks=(10,),
            )

        with self.assertRaisesRegex(BenchmarkValidationError, "empty benchmark"):
            evaluate_retrieval_route(
                (),
                lambda request, limit: (),
                valid_ids=self.catalog.ids,
                ks=(10,),
            )

        class Valid:
            parent_asin = "A_HIKE"

        with self.assertRaisesRegex(BenchmarkValidationError, "more than the requested"):
            evaluate_retrieval_route(
                cases[:1],
                lambda request, limit: [Valid(), Valid()],
                valid_ids=self.catalog.ids,
                ks=(1,),
            )

    def test_rejects_target_leakage_across_tuning_and_heldout(self) -> None:
        first = json.loads(CASES.read_text(encoding="utf-8").splitlines()[0])
        leaked = {**first, "case_id": "leaked-heldout", "split": "heldout"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaked.jsonl"
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(leaked) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BenchmarkValidationError, "target leakage"):
                load_retrieval_benchmark_cases(
                    path,
                    valid_ids=self.catalog.ids,
                    require_all_scenarios_per_split=False,
                )

    def test_diagnostic_summary_is_split_aware_and_normalizes_small_catalog_overlap(self) -> None:
        cases = load_retrieval_benchmark_cases(CASES, valid_ids=self.catalog.ids)
        records = [
            {
                "sparse_candidates": 8,
                "dense_candidates": 8,
                "fused_candidates": 8,
                "hard_filtered_candidates": 0,
                "returned_candidates": 5,
                "route_overlap": {10: 8},
                "route_timings_ms": {"sparse": 2.0, "dense": 1.0, "total": 4.0},
            }
            for _ in cases
        ]

        report = _diagnostic_report(cases, records)

        heldout = report["splits"]["heldout"]
        self.assertEqual(heldout["overall"]["case_count"], 4)
        self.assertEqual(heldout["overall"]["mean_route_overlap_rate"]["10"], 1.0)
        self.assertEqual(
            set(heldout["scenarios"]),
            {"buying", "browsing", "intent_override", "boundary"},
        )


if __name__ == "__main__":
    unittest.main()
