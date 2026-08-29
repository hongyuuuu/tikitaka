from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from tikitaka.retrieval.catalog import CatalogValidationError, load_catalog


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class CatalogTest(unittest.TestCase):
    def test_loads_immutable_catalog_with_stable_identity(self) -> None:
        before = FIXTURE.read_bytes()
        catalog = load_catalog(FIXTURE, expected_count=7)

        self.assertEqual(len(catalog), 7)
        self.assertEqual(catalog.identity.row_count, 7)
        self.assertEqual(len(catalog.identity.source_sha256), 64)
        self.assertEqual(len(catalog.identity.ordered_parent_asin_sha256), 64)
        self.assertEqual(FIXTURE.read_bytes(), before)
        self.assertEqual(catalog.require("A_HIKE").price, Decimal("79.99"))
        self.assertEqual(
            tuple(key for key, _ in catalog.require("A_HIKE").details),
            ("Color", "Manufacturer", "Size", "Style"),
        )
        with self.assertRaises(TypeError):
            catalog.by_id["X"] = catalog.require("A_HIKE")  # type: ignore[index]

    def test_rejects_duplicate_missing_blank_and_malformed_records(self) -> None:
        good = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
        cases = {
            "duplicate": json.dumps(good) + "\n" + json.dumps(good) + "\n",
            "missing_id": json.dumps({"title": "x"}) + "\n",
            "blank": json.dumps(good) + "\n\n",
            "malformed": "{not-json}\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "catalog.jsonl"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(CatalogValidationError):
                    load_catalog(path)

    def test_expected_count_is_enforced(self) -> None:
        with self.assertRaisesRegex(CatalogValidationError, "row count mismatch"):
            load_catalog(FIXTURE, expected_count=50_000)

    def test_catalog_price_sentinels_remain_unknown(self) -> None:
        rows = (
            '{"parent_asin":"DASH","price":"—"}\n'
            '{"parent_asin":"VARIANT","price":"from 12.99"}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(rows, encoding="utf-8")
            catalog = load_catalog(path)

        self.assertIsNone(catalog.require("DASH").price)
        self.assertIsNone(catalog.require("VARIANT").price)


if __name__ == "__main__":
    unittest.main()
