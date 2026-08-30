"""Fake-only reconciliation tests; no credential or network is used."""

from __future__ import annotations

import unittest

from tikitaka.contracts import Usage
from tikitaka.evaluation.accounting import ProviderUsageSnapshot, reconcile_usage
from tikitaka.models.usage import merge


class UsageReconciliationTests(unittest.TestCase):
    def _events(self) -> tuple[tuple[str, Usage], ...]:
        interpreter = Usage(
            prompt_tokens=100,
            completion_tokens=40,
            reasoning_tokens=20,
            calls=1,
            estimated_cost=0.00068,
        )
        repair = Usage(
            prompt_tokens=120,
            completion_tokens=30,
            reasoning_tokens=10,
            calls=1,
            repairs=1,
            estimated_cost=0.00060,
        )
        reranker = Usage(
            prompt_tokens=80,
            completion_tokens=20,
            reasoning_tokens=5,
            calls=1,
            estimated_cost=0.00040,
        )
        return (("interpreter", merge(interpreter, repair)), ("reranker", reranker))

    def test_every_component_and_repair_reconciles_to_an_isolated_snapshot(self) -> None:
        report = reconcile_usage(
            self._events(),
            ProviderUsageSnapshot(300, 90, billed_cost=0.00168),
        )
        self.assertEqual(report["status"], "matched")
        self.assertEqual(report["local"]["calls"], 3)
        self.assertEqual(report["local"]["repairs"], 1)
        self.assertEqual(report["by_component"]["interpreter"]["prompt_tokens"], 220)
        self.assertEqual(report["by_component"]["reranker"]["completion_tokens"], 20)

    def test_missing_provider_tokens_are_reported_not_explained_away(self) -> None:
        report = reconcile_usage(
            self._events(),
            ProviderUsageSnapshot(500, 140, billed_cost=0.004),
        )
        self.assertEqual(report["status"], "local_underreported")
        self.assertEqual(
            report["delta_provider_minus_local"],
            {"prompt_tokens": 200, "completion_tokens": 50, "cost": 0.00232},
        )

    def test_shared_day_billing_is_not_treated_as_a_reconcilable_snapshot(self) -> None:
        report = reconcile_usage(self._events())
        self.assertEqual(report["status"], "provider_snapshot_required")
        self.assertIsNone(report["provider"])
        self.assertIsNone(report["delta_provider_minus_local"])

    def test_invalid_provider_totals_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            ProviderUsageSnapshot(-1, 0)
        with self.assertRaises(ValueError):
            reconcile_usage(self._events(), cost_tolerance=-1)


if __name__ == "__main__":
    unittest.main()
