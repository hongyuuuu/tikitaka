from __future__ import annotations

import unittest
from pathlib import Path

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.text import (
    DENSE_QUERY_SCHEMA_VERSION,
    PRODUCT_TEXT_SCHEMA_VERSION,
    build_dense_query,
    build_dense_text,
    build_sparse_fields,
    fts5_expression,
    normalize_text,
    query_terms,
)
from tikitaka.retrieval.request import RetrievalConstraint


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class RetrievalTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.product = load_catalog(FIXTURE).require("A_HIKE")

    def test_product_text_v1_is_deterministic_and_labeled(self) -> None:
        first = build_dense_text(self.product)
        second = build_dense_text(self.product)
        self.assertEqual(PRODUCT_TEXT_SCHEMA_VERSION, "product_text_v1")
        self.assertEqual(first, second)
        self.assertIn("TITLE: Blue Leather Waterproof Hiking Boots", first)
        self.assertIn("CATEGORY: Women > Shoes > Boots", first)
        self.assertNotIn("CATEGORY: Clothing, Shoes & Jewelry", first)
        self.assertIn("DETAILS: Color: Blue | Manufacturer: TrailCo | Size: 8 | Style: Hiking", first)
        self.assertIn("PRICE: 79.99", first)

    def test_sparse_fields_remain_separate_for_bm25_weights(self) -> None:
        fields = build_sparse_fields(self.product)
        self.assertEqual(fields.title, "Blue Leather Waterproof Hiking Boots")
        self.assertEqual(fields.categories, "Women > Shoes > Boots")
        self.assertIn("Water-resistant membrane", fields.features)
        self.assertEqual(len(fields.ordered_values()), 6)

    def test_query_tokenization_is_safe_stable_and_size_aware(self) -> None:
        self.assertEqual(normalize_text("  café\t boots  "), "café boots")
        self.assertEqual(query_terms("I need BOOTS boots in size S"), ("need", "boots", "size", "s"))
        expression = fts5_expression(text_query='boots" OR *', should_terms=("waterproof",))
        self.assertEqual(expression, '"boots" OR "waterproof"')

    def test_dense_query_contains_only_current_positive_structured_intent(self) -> None:
        query = build_dense_query(
            "comfortable travel shoes",
            must_terms=("walking",),
            should_terms=("water resistant",),
            constraints=(
                RetrievalConstraint("category", ("shoes",), strength="hard"),
                RetrievalConstraint(
                    "material",
                    ("leather",),
                    polarity="exclude",
                    strength="hard",
                ),
                RetrievalConstraint("budget", (80,), strength="hard", operator="lte"),
            ),
        )
        self.assertEqual(DENSE_QUERY_SCHEMA_VERSION, "dense_query_v1")
        self.assertIn("QUERY: comfortable travel shoes", query)
        self.assertIn("CONSTRAINT_CATEGORY: shoes", query)
        self.assertIn("CONSTRAINT_BUDGET: 80", query)
        self.assertNotIn("leather", query)


if __name__ == "__main__":
    unittest.main()
