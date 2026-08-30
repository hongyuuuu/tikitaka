from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from scripts.run_m5_offline import (
    NetworkAuditGuard,
    ObservedAgent,
    OfflineEvidenceError,
    credential_absent,
    probe_dense_artifact_failures,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "fixtures" / "catalog_small.jsonl"


class _Delegate:
    route_id = "heuristic/local"

    class _ShoppingAgent:
        degraded = True

    _shopping_agent = _ShoppingAgent()

    def reset(self, _session_id: str, _profile: dict) -> None:
        return None

    def respond(self, *_args: object) -> dict:
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A_HIKE"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def close(self) -> None:
        return None


class M5OfflineEvidenceTests(unittest.TestCase):
    def test_network_guard_denies_and_records_dns(self) -> None:
        guard = NetworkAuditGuard()
        import sys

        sys.addaudithook(guard)
        with self.assertRaisesRegex(OfflineEvidenceError, "network access denied"):
            with guard.deny():
                socket.getaddrinfo("example.invalid", 443)
        self.assertEqual(guard.attempts["socket.getaddrinfo"], 1)
        self.assertFalse(guard.active)

    def test_credential_is_absent_only_inside_context(self) -> None:
        name = "TIKITAKA_M5_TEST_CREDENTIAL"
        os.environ[name] = "secret-value"
        try:
            with credential_absent(name):
                self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ[name], "secret-value")
        finally:
            os.environ.pop(name, None)

    def test_observer_accepts_valid_unique_catalog_ids(self) -> None:
        observed = ObservedAgent(_Delegate(), {"A_HIKE"})  # type: ignore[arg-type]
        response = observed.respond("session", "message", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A_HIKE")
        self.assertEqual(observed.response_count, 1)
        self.assertEqual(observed.exceptions, {})
        self.assertEqual(observed.violations, {})
        self.assertTrue(observed.degraded)

    def test_observer_records_raw_contract_violations(self) -> None:
        class Invalid(_Delegate):
            def respond(self, *_args: object) -> dict:
                return {
                    "message": "bad",
                    "ask_attribute": "material",
                    "recommendations": [
                        {"parent_asin": "A_HIKE"},
                        {"parent_asin": "A_HIKE"},
                        {"parent_asin": "NOT_IN_CATALOG"},
                    ],
                    "usage": {"prompt_tokens": -1, "completion_tokens": True},
                }

        observed = ObservedAgent(Invalid(), {"A_HIKE"})  # type: ignore[arg-type]
        observed.respond("session", "message", 1, 10)
        self.assertEqual(observed.violations["clarify_with_recommendations"], 1)
        self.assertEqual(observed.violations["duplicate_catalog_id"], 1)
        self.assertEqual(observed.violations["invalid_catalog_id"], 1)
        self.assertEqual(observed.violations["invalid_prompt_tokens"], 1)
        self.assertEqual(observed.violations["invalid_completion_tokens"], 1)

    def test_dense_artifact_failures_are_nonfatal_and_coded(self) -> None:
        checks = probe_dense_artifact_failures(CATALOG)
        self.assertEqual(
            checks,
            {
                "missing_artifact": "dense_artifact_unavailable",
                "corrupt_artifact": "dense_artifact_invalid",
            },
        )


if __name__ == "__main__":
    unittest.main()
