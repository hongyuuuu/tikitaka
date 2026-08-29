from __future__ import annotations

import unittest
from pathlib import Path

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.sparse import SparseIndex


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class SparseRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(FIXTURE)
        cls.index = SparseIndex(cls.catalog)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index.close()

    def test_exact_lexical_match_is_ranked_first(self) -> None:
        hits = self.index.search("waterproof leather hiking boots", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].parent_asin, "A_HIKE")
        self.assertEqual(hits[0].rank, 1)
        self.assertIn("title", hits[0].matched_fields)

    def test_must_terms_and_empty_queries_are_handled(self) -> None:
        hits = self.index.search("athletic shoe", must_terms=("running",), limit=5)
        self.assertEqual(hits[0].parent_asin, "B_RUN")
        self.assertEqual(self.index.search("", limit=5), [])

    def test_order_is_deterministic_and_ids_are_unique(self) -> None:
        first = self.index.search("travel work comfort", limit=7)
        second = self.index.search("travel work comfort", limit=7)
        self.assertEqual(first, second)
        identifiers = [hit.parent_asin for hit in first]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(set(identifiers).issubset(self.catalog.ids))

    def test_manifest_couples_index_to_catalog_and_text_version(self) -> None:
        manifest = self.index.manifest
        self.assertEqual(manifest.catalog_row_count, 7)
        self.assertEqual(manifest.catalog_sha256, self.catalog.identity.source_sha256)
        self.assertEqual(manifest.product_text_schema_version, "product_text_v1")


if __name__ == "__main__":
    unittest.main()
