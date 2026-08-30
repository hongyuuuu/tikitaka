from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
import stat
from pathlib import Path

from scripts.m6_release import (
    ReleaseAuditError,
    audit_submission_archive,
    audit_tracked_repository,
    build_submission,
)
from scripts.capture_m6_retrieval_traces import trace_requests


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.jsonl"


class M6ReleaseTests(unittest.TestCase):
    def test_tracked_repository_passes_artifact_policy(self) -> None:
        report = audit_tracked_repository()
        self.assertEqual(report["forbidden_tracked_files"], [])
        self.assertEqual(report["oversized_tracked_files"], [])

    @unittest.skipUnless(CATALOG.is_file(), "full frozen catalog is not available")
    def test_build_is_manifested_and_excludes_organizer_material(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tikitaka-m6-test-") as directory:
            output = Path(directory) / "submission.zip"
            report = build_submission(output, catalog=CATALOG, allow_dirty=True)
            repeated_output = Path(directory) / "submission-repeated.zip"
            repeated = build_submission(
                repeated_output,
                catalog=CATALOG,
                allow_dirty=True,
            )
            audited = audit_submission_archive(output)
            self.assertEqual(report["archive_sha256"], audited["archive_sha256"])
            self.assertEqual(report["archive_sha256"], repeated["archive_sha256"])
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("manifest.json"))
            self.assertIn("agent.py", names)
            self.assertIn("REPORT.md", names)
            self.assertIn("starter/agent.py", names)
            self.assertIn("tikitaka/retrieval/hybrid.py", names)
            self.assertNotIn("data/catalog.jsonl", names)
            self.assertNotIn("data/public_set.jsonl", names)
            self.assertFalse(any(name.startswith("evaluator/") for name in names))
            self.assertFalse(any(name.startswith("tests/") for name in names))
            self.assertFalse(any(name.startswith("reports/") for name in names))
            self.assertFalse(manifest["catalog"]["included"])
            self.assertFalse(manifest["runtime"]["network_required"])
            self.assertEqual(
                manifest["frozen_retrieval_config"]["sparse"]["field_weights"]
                if "field_weights" in manifest["frozen_retrieval_config"]["sparse"]
                else [
                    manifest["frozen_retrieval_config"]["sparse"][name]
                    for name in (
                        "title_weight",
                        "category_weight",
                        "feature_weight",
                        "detail_weight",
                        "store_weight",
                        "description_weight",
                    )
                ],
                [6.0, 4.0, 2.5, 2.5, 1.5, 1.0],
            )

    def test_override_trace_uses_only_the_reduced_new_intent(self) -> None:
        requests = trace_requests()
        before = requests["intent_override_before"]
        after = requests["intent_override_after"]
        self.assertEqual(before.intent_version, 1)
        self.assertEqual(after.intent_version, 2)
        self.assertEqual(after.no_preference, frozenset({"color"}))
        self.assertEqual(after.profile_terms, ())
        self.assertEqual(after.profile_weight, 0.0)
        after_values = {
            str(value).casefold()
            for constraint in after.constraints
            for value in constraint.values
        }
        self.assertNotIn("leather", after_values)
        self.assertNotIn("red", after_values)
        self.assertNotIn("boots", after_values)

    def test_archive_audit_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tikitaka-m6-test-") as directory:
            output = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("agent.py", "class Agent: pass\n")
                archive.writestr("manifest.json", '{"files": {}}\n')
                archive.writestr("../secret", "nope")
            with self.assertRaisesRegex(ReleaseAuditError, "unsafe archive path"):
                audit_submission_archive(output)

    def test_archive_audit_rejects_secret_material(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tikitaka-m6-test-") as directory:
            output = Path(directory) / "secret.zip"
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("agent.py", "token = 'sk-proj-abcdefghijklmnopqrstuvwxyz'\n")
                archive.writestr("manifest.json", '{"files": {}}\n')
            with self.assertRaisesRegex(ReleaseAuditError, "possible secret"):
                audit_submission_archive(output)

    def test_archive_audit_rejects_symlink_and_root_file_masquerading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tikitaka-m6-test-") as directory:
            symlink_archive = Path(directory) / "symlink.zip"
            with zipfile.ZipFile(symlink_archive, "w") as archive:
                link = zipfile.ZipInfo("tikitaka/link.py")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "../../secret")
                archive.writestr("agent.py", "class Agent: pass\n")
                archive.writestr("manifest.json", '{"files": {}}\n')
            with self.assertRaisesRegex(ReleaseAuditError, "symlink is forbidden"):
                audit_submission_archive(symlink_archive)

            masquerade_archive = Path(directory) / "masquerade.zip"
            with zipfile.ZipFile(masquerade_archive, "w") as archive:
                archive.writestr("agent.py/hidden.py", "pass\n")
                archive.writestr("agent.py", "class Agent: pass\n")
                archive.writestr("manifest.json", '{"files": {}}\n')
            with self.assertRaisesRegex(ReleaseAuditError, "root file used as directory"):
                audit_submission_archive(masquerade_archive)


if __name__ == "__main__":
    unittest.main()
