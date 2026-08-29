"""Person 1 state tests. Offline, no catalog, no model, no network."""

from __future__ import annotations

import unittest

from tikitaka.contracts.domain import StateDelta, StateOperation
from tikitaka.models.fake import FaultyInterpreter, HeuristicInterpreter
from tikitaka.state.extractor import Extractor
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import SCHEMA_VERSION, parse
from tikitaka.state.session import SessionState, new_session

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


def add(attribute: str, value: object, strength: str = "soft") -> StateOperation:
    return StateOperation(
        operation="add",
        attribute=attribute,
        new_value=value,
        polarity="include",
        strength=strength,
        confidence=0.8,
    )


def delta(*operations: StateOperation, mode: str = "buying") -> StateDelta:
    return StateDelta(
        inferred_mode=mode,
        mode_confidence=0.7,
        operations=operations,
        generality=0.4,
        schema_version=SCHEMA_VERSION,
    )


def values(state: SessionState, attribute: str) -> set[object]:
    return {c.normalized_value for c in state.constraints_for(attribute)}


class ReducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reducer = StateReducer()
        self.state = new_session("s1", PROFILE)

    def test_accumulates_compatible_constraints(self) -> None:
        self.reducer.apply(
            self.state,
            delta(add("category", "running shoes"), add("color", "blue")),
            1,
        )
        self.reducer.apply(self.state, delta(add("budget", "under $80")), 2)
        active = {c.attribute for c in self.state.active_constraints}
        self.assertEqual(active, {"category", "color", "budget"})

    def test_replace_conflicting_color(self) -> None:
        self.reducer.apply(self.state, delta(add("color", "blue")), 1)
        self.reducer.apply(self.state, delta(add("color", "red")), 2)
        self.assertEqual(values(self.state, "color"), {"red"})

    def test_replace_conflicting_material(self) -> None:
        self.reducer.apply(self.state, delta(add("material", "cotton")), 1)
        self.reducer.apply(self.state, delta(add("material", "wool")), 2)
        self.assertEqual(values(self.state, "material"), {"wool"})

    def test_replace_conflicting_category(self) -> None:
        self.reducer.apply(self.state, delta(add("category", "boots")), 1)
        self.reducer.apply(self.state, delta(add("category", "sandals")), 2)
        self.assertEqual(values(self.state, "category"), {"sandals"})

    def test_repeated_value_refreshes_rather_than_duplicating(self) -> None:
        self.reducer.apply(self.state, delta(add("color", "blue")), 1)
        self.reducer.apply(self.state, delta(add("color", "blue")), 3)
        self.assertEqual(len(self.state.constraints_for("color")), 1)
        self.assertEqual(self.state.constraints_for("color")[0].source_turn, 3)

    def test_remove_budget_does_not_create_negative_budget(self) -> None:
        self.reducer.apply(self.state, delta(add("budget", "under $50")), 1)
        self.reducer.apply(
            self.state, delta(StateOperation(operation="remove", attribute="budget")), 2
        )
        self.assertEqual(self.state.constraints_for("budget"), ())
        for constraint in self.state.constraint_history:
            self.assertNotEqual(constraint.polarity, "exclude")
            if constraint.attribute == "budget":
                self.assertGreaterEqual(constraint.normalized_value, 0)

    def test_exclude_leather_and_exclude_red(self) -> None:
        self.reducer.apply(self.state, delta(add("material", "leather")), 1)
        self.reducer.apply(
            self.state,
            delta(
                StateOperation(
                    operation="exclude",
                    attribute="material",
                    new_value="leather",
                    polarity="exclude",
                    strength="hard",
                    confidence=0.9,
                ),
                StateOperation(
                    operation="exclude",
                    attribute="color",
                    new_value="red",
                    polarity="exclude",
                    strength="soft",
                    confidence=0.7,
                ),
            ),
            2,
        )
        materials = self.state.constraints_for("material")
        self.assertEqual(len(materials), 1)
        self.assertEqual(materials[0].polarity, "exclude")
        self.assertEqual(self.state.constraints_for("color")[0].polarity, "exclude")

    def test_no_preference_suppresses_future_asks(self) -> None:
        self.reducer.apply(
            self.state,
            delta(StateOperation(operation="no_preference", attribute="material")),
            1,
        )
        self.assertIn("material", self.state.no_preference)
        self.assertFalse(self.state.is_askable("material"))
        self.assertEqual(self.state.constraints_for("material"), ())

    def test_full_reset_versus_single_attribute_replacement(self) -> None:
        self.reducer.apply(
            self.state, delta(add("category", "boots"), add("color", "brown")), 1
        )
        version = self.state.intent_version

        self.reducer.apply(self.state, delta(add("color", "black")), 2)
        self.assertEqual(self.state.intent_version, version)
        self.assertEqual(values(self.state, "category"), {"boots"})

        self.reducer.apply(
            self.state,
            delta(StateOperation(operation="reset", scope="conversation")),
            3,
        )
        self.assertEqual(self.state.active_constraints, ())
        self.assertEqual(self.state.intent_version, version + 1)
        self.assertEqual(dict(self.state.profile_seed), PROFILE)

    def test_intent_scope_reset_preserves_budget_but_conversation_scope_does_not(
        self,
    ) -> None:
        for scope, expected in (("intent", {80.0}), ("conversation", set())):
            with self.subTest(scope=scope):
                state = new_session(f"s-{scope}", PROFILE)
                self.reducer.apply(
                    state,
                    delta(add("category", "boots"), add("budget", "under $80")),
                    1,
                )
                self.reducer.apply(
                    state,
                    delta(StateOperation(operation="reset", scope=scope)),
                    2,
                )
                self.assertEqual(values(state, "budget"), expected)

    def test_category_change_clears_dependent_preserves_budget_flags_ambiguous(
        self,
    ) -> None:
        self.reducer.apply(
            self.state,
            delta(
                add("category", "hiking boots"),
                add("budget", "under $120"),
                add("material", "leather"),
                add("size", "size 10"),
                add("color", "brown"),
            ),
            1,
        )
        version = self.state.intent_version
        self.reducer.apply(self.state, delta(add("category", "wool sweaters")), 2)

        self.assertEqual(self.state.intent_version, version + 1)
        self.assertEqual(values(self.state, "budget"), {120.0})
        self.assertEqual(self.state.constraints_for("material"), ())
        self.assertEqual(self.state.constraints_for("size"), ())
        flagged = {c.attribute for c in self.state.revalidation_constraints}
        self.assertIn("color", flagged)
        self.assertEqual(values(self.state, "category"), {"wool sweaters"})

    def test_new_intent_version_reopens_shown_products(self) -> None:
        from tikitaka.contracts.domain import TurnDecision

        self.reducer.apply(self.state, delta(add("category", "boots")), 1)
        self.reducer.record_decision(
            self.state,
            TurnDecision(
                action="recommend", ask_attribute=None, reason_code="ranking_stable"
            ),
            ["B001", "B002"],
        )
        self.assertEqual(self.state.shown_product_ids, frozenset({"B001", "B002"}))
        self.reducer.apply(self.state, delta(add("category", "sandals")), 2)
        self.assertEqual(self.state.shown_product_ids, frozenset())

    def test_clarify_records_asked_attribute(self) -> None:
        from tikitaka.contracts.domain import TurnDecision

        self.reducer.record_decision(
            self.state,
            TurnDecision(
                action="clarify",
                ask_attribute="material",
                reason_code="valuable_clarification",
            ),
        )
        self.assertIn("material", self.state.asked_attributes)
        self.assertFalse(self.state.is_askable("material"))

    def test_attribute_exhaustion_is_distinct_from_no_preference(self) -> None:
        self.reducer.note_exhausted(self.state, "brand")
        self.assertIn("brand", self.state.exhausted_attributes)
        self.assertNotIn("brand", self.state.no_preference)
        self.assertFalse(self.state.is_askable("brand"))

    def test_reducer_is_order_stable_for_a_multi_operation_delta(self) -> None:
        self.reducer.apply(self.state, delta(add("color", "blue")), 1)
        override = delta(
            add("budget", "under $60"),
            StateOperation(
                operation="replace",
                attribute="color",
                old_value="blue",
                new_value="green",
                polarity="include",
                strength="hard",
                confidence=0.9,
            ),
        )
        self.reducer.apply(self.state, override, 2)
        self.assertEqual(values(self.state, "color"), {"green"})
        self.assertEqual(values(self.state, "budget"), {60.0})

    def test_sessions_do_not_share_state(self) -> None:
        first = new_session("a", PROFILE)
        second = new_session("b", PROFILE)
        self.reducer.apply(first, delta(add("color", "blue")), 1)
        self.assertEqual(second.active_constraints, ())
        self.assertEqual(second.shown_product_ids, frozenset())
        first.profile_seed["preference_tags"].append("mutated")  # type: ignore[union-attr]
        self.assertEqual(
            dict(second.profile_seed)["preference_tags"], PROFILE["preference_tags"]
        )

    def test_explicit_dialogue_state_outranks_profile(self) -> None:
        self.reducer.apply(self.state, delta(add("material", "nylon")), 1)
        attributes = {c.attribute for c in self.state.active_constraints}
        self.assertEqual(attributes, {"material"})
        for constraint in self.state.active_constraints:
            self.assertNotIn(constraint.normalized_value, PROFILE["preference_tags"])


class ExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = new_session("s2", PROFILE)
        self.extractor = Extractor(interpreter=HeuristicInterpreter())

    def test_buying_opening_extracts_category_and_requirement(self) -> None:
        self.extractor.ingest(
            self.state,
            "I'm looking for running shoes. A key requirement is: waterproof leather.",
            1,
        )
        self.assertEqual(self.state.mode, "buying")
        attributes = {c.attribute for c in self.state.active_constraints}
        self.assertIn("category", attributes)
        self.assertIn("material", attributes)

    def test_browsing_opening_sets_high_generality(self) -> None:
        self.extractor.ingest(
            self.state, "I'm looking for sweaters, but I'm still exploring.", 1
        )
        self.assertEqual(self.state.mode, "browsing")
        self.assertGreater(self.state.generality, 0.7)

    def test_boundary_reply_records_no_preference(self) -> None:
        self.extractor.ingest(
            self.state,
            "I don't have a preference for material; please use your judgment.",
            2,
        )
        self.assertIn("material", self.state.no_preference)
        self.assertNotIn("material", self.state.exhausted_attributes)

    def test_exhaustion_reply_is_not_no_preference(self) -> None:
        result = self.extractor.ingest(
            self.state, "I don't have an additional preference for brand.", 2
        )
        self.assertEqual(result.exhausted_attribute, "brand")
        self.assertIn("brand", self.state.exhausted_attributes)
        self.assertNotIn("brand", self.state.no_preference)

    def test_model_failure_degrades_to_heuristic_route(self) -> None:
        extractor = Extractor(interpreter=FaultyInterpreter(mode="timeout"))
        result = extractor.ingest(
            self.state, "I'm looking for boots. A key requirement is: wool lining.", 1
        )
        self.assertTrue(result.used_fallback)
        self.assertTrue(self.state.active_constraints)

    def test_component_exception_never_escapes(self) -> None:
        extractor = Extractor(interpreter=FaultyInterpreter(mode="exception"))
        result = extractor.ingest(self.state, "I'm looking for hats.", 1)
        self.assertTrue(result.failure)
        self.assertIsInstance(result.usage.prompt_tokens, int)


class ParserFailureTests(unittest.TestCase):
    def test_rejects_unknown_attributes_and_operations(self) -> None:
        result = parse(
            '{"inferred_mode": "buying", "operations": ['
            '{"operation": "obliterate", "attribute": "color"},'
            '{"operation": "add", "attribute": "vibe", "new_value": "x",'
            ' "polarity": "include", "strength": "soft", "confidence": 0.5},'
            '{"operation": "add", "attribute": "color", "new_value": "red",'
            ' "polarity": "include", "strength": "soft", "confidence": 0.5}]}'
        )
        self.assertEqual(result.delta.rejected_operations, 2)
        self.assertEqual(len(result.delta.operations), 1)
        self.assertEqual(result.delta.operations[0].attribute, "color")

    def test_clamps_invalid_confidence(self) -> None:
        result = parse(
            '{"inferred_mode": "buying", "mode_confidence": 5, "generality": -2,'
            ' "operations": [{"operation": "add", "attribute": "color",'
            ' "new_value": "red", "polarity": "include", "strength": "soft",'
            ' "confidence": 99}]}'
        )
        self.assertEqual(result.delta.mode_confidence, 1.0)
        self.assertEqual(result.delta.generality, 0.0)
        self.assertEqual(result.delta.operations[0].confidence, 1.0)

    def test_malformed_json_recovers_or_fails_safely(self) -> None:
        for payload in ("", "not json", '{"operations": [', "[1, 2, 3]", None):
            with self.subTest(payload=payload):
                result = parse(payload)
                self.assertTrue(result.top_level_failure)
                self.assertEqual(result.delta.operations, ())

    def test_json_wrapped_in_prose_is_recovered(self) -> None:
        result = parse(
            'Sure! Here you go:\n{"inferred_mode": "browsing", "operations": []}\nDone.'
        )
        self.assertFalse(result.top_level_failure)
        self.assertEqual(result.delta.inferred_mode, "browsing")


if __name__ == "__main__":
    unittest.main()
