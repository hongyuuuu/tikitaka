from __future__ import annotations

import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.request import request_from_search_plan
from tikitaka.retrieval.structured import evaluate_constraint, extract_structured_evidence


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class StructuredRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(FIXTURE)

    def test_extracts_high_confidence_values_with_provenance(self) -> None:
        evidence = extract_structured_evidence(self.catalog.require("A_HIKE"))
        self.assertIn("leather", evidence.for_attribute("material").values)
        self.assertIn("blue", evidence.for_attribute("color").values)
        self.assertIn("trailco", evidence.for_attribute("brand").values)
        self.assertEqual(evidence.for_attribute("budget").values, (Decimal("79.99"),))
        self.assertIn("shoes", evidence.for_attribute("category").values)
        self.assertNotIn("clothing, shoes & jewelry", evidence.for_attribute("category").values)
        self.assertGreaterEqual(evidence.for_attribute("material").reliability, 0.9)

    def test_budget_known_and_missing_are_not_conflated(self) -> None:
        known = extract_structured_evidence(self.catalog.require("A_HIKE"))
        missing = extract_structured_evidence(self.catalog.require("C_WALK_UNKNOWN"))
        under_eighty = evaluate_constraint(
            known, attribute="budget", desired_values=(80,), operator="lte"
        )
        under_sixty = evaluate_constraint(
            known, attribute="budget", desired_values=(60,), operator="lte"
        )
        missing_price = evaluate_constraint(
            missing, attribute="budget", desired_values=(60,), operator="lte"
        )
        self.assertEqual(under_eighty.outcome, "match")
        self.assertEqual(under_sixty.outcome, "contradiction")
        self.assertEqual(missing_price.outcome, "unknown")

    def test_exclusion_requires_positive_evidence(self) -> None:
        leather = extract_structured_evidence(self.catalog.require("A_HIKE"))
        unspecified = extract_structured_evidence(self.catalog.require("C_WALK_UNKNOWN"))
        self.assertEqual(
            evaluate_constraint(
                leather,
                attribute="material",
                desired_values=("leather",),
                polarity="exclude",
            ).outcome,
            "contradiction",
        )
        self.assertEqual(
            evaluate_constraint(
                unspecified,
                attribute="material",
                desired_values=("leather",),
                polarity="exclude",
            ).outcome,
            "unknown",
        )

    def test_search_plan_adapter_preserves_filters_revalidation_and_profile(self) -> None:
        plan = SimpleNamespace(
            text_query="comfortable shoes",
            must_terms=("shoes",),
            should_terms=("comfort",),
            exclude_terms=("leather",),
            filters={
                "budget": {"max": 80},
                "material": {"value": "leather", "polarity": "exclude"},
            },
            attribute_values={"budget": (80,), "material": ("leather",)},
            mode="buying",
            intent_version=2,
            revalidation_flags=frozenset({"material"}),
            no_preference=frozenset(),
            profile_bias=SimpleNamespace(terms=("blue",), weight=0.2),
        )
        request = request_from_search_plan(plan)
        self.assertEqual(request.intent_version, 2)
        self.assertEqual(request.profile_terms, ("blue",))
        self.assertEqual(request.profile_weight, 0.2)
        constraints = {item.attribute: item for item in request.constraints}
        self.assertEqual(constraints["budget"].operator, "lte")
        self.assertEqual(constraints["material"].polarity, "exclude")
        self.assertEqual(constraints["material"].strength, "soft")
        self.assertTrue(constraints["material"].needs_revalidation)


if __name__ == "__main__":
    unittest.main()
