"""Person 1 query-builder tests, including the DG-02 profile ablation proof."""

from __future__ import annotations

import unittest

from tikitaka.retrieval.request import request_from_search_plan
from tikitaka.state.query_builder import ActiveQueryBuilder, QueryBuilderConfig
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import make_delta, operation
from tikitaka.state.session import new_session

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


def add(attribute: str, value: object, strength: str = "soft"):
    return operation(
        "add",
        attribute=attribute,
        new_value=value,
        polarity="include",
        strength=strength,
        confidence=0.8,
    )


def delta(*operations, mode: str = "buying"):
    return make_delta(
        inferred_mode=mode,
        mode_confidence=0.7,
        operations=operations,
        generality=0.4,
    )


class QueryBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reducer = StateReducer()
        self.builder = ActiveQueryBuilder()

    def state_with(self, *operations, profile=PROFILE, turn: int = 1):
        state = new_session("qb", profile)
        self.reducer.apply(state, delta(*operations), turn)
        return state

    def test_empty_state_produces_a_valid_plan(self) -> None:
        plan = self.builder.build(new_session("empty", PROFILE))
        self.assertEqual(plan.text_query, "")
        self.assertEqual(plan.must_terms, ())
        self.assertEqual(plan.intent_version, 1)

    def test_hard_and_soft_constraints_split_correctly(self) -> None:
        state = self.state_with(
            add("category", "hiking boots", strength="hard"),
            add("color", "brown", strength="soft"),
        )
        plan = self.builder.build(state)
        self.assertIn("hiking boots", plan.must_terms)
        self.assertIn("brown", plan.should_terms)
        self.assertNotIn("brown", plan.must_terms)

    def test_exclusions_are_structured_not_buried_in_text(self) -> None:
        state = self.state_with(add("material", "leather", strength="hard"))
        self.reducer.apply(
            state,
            delta(
                operation(
                    "exclude",
                    attribute="material",
                    new_value="leather",
                    polarity="exclude",
                    strength="hard",
                    confidence=0.9,
                )
            ),
            2,
        )
        plan = self.builder.build(state)
        self.assertIn("leather", plan.exclude_terms)
        self.assertNotIn("leather", plan.should_terms)
        self.assertNotIn("leather", plan.must_terms)
        self.assertNotIn("leather", plan.text_query)

    def test_budget_becomes_a_numeric_filter_not_a_text_term(self) -> None:
        state = self.state_with(add("budget", "under $80", strength="hard"))
        plan = self.builder.build(state)
        self.assertEqual(plan.filters["budget"]["max"], 80.0)
        self.assertNotIn("80", plan.text_query)
        self.assertEqual(plan.must_terms, ())

    def test_plan_is_deterministic_for_fixed_state(self) -> None:
        state = self.state_with(
            add("category", "boots", strength="hard"),
            add("color", "brown"),
            add("use_case", "hiking"),
        )
        first = self.builder.build(state)
        second = self.builder.build(state)
        self.assertEqual(first, second)
        self.assertEqual(first.text_query, second.text_query)

    def test_retracted_constraints_never_reach_the_plan(self) -> None:
        state = self.state_with(add("color", "blue"))
        self.reducer.apply(state, delta(add("color", "green")), 2)
        plan = self.builder.build(state)
        self.assertNotIn("blue", plan.text_query)
        self.assertIn("green", plan.should_terms)

    def test_new_intent_version_drops_old_dependent_constraints(self) -> None:
        state = self.state_with(
            add("category", "boots", strength="hard"),
            add("material", "leather", strength="hard"),
            add("budget", "under $90", strength="hard"),
        )
        self.reducer.apply(state, delta(add("category", "sweaters", strength="hard")), 2)
        plan = self.builder.build(state)
        self.assertEqual(plan.intent_version, 2)
        self.assertNotIn("leather", plan.text_query)
        self.assertEqual(plan.filters["budget"]["max"], 90.0)

    def test_revalidation_constraints_are_soft_and_flagged(self) -> None:
        state = self.state_with(
            add("category", "boots", strength="hard"),
            add("color", "brown", strength="hard"),
        )
        self.reducer.apply(state, delta(add("category", "sweaters", strength="hard")), 2)
        plan = self.builder.build(state)
        self.assertIn("color", plan.revalidation_flags)
        self.assertIn("brown", plan.should_terms)
        self.assertNotIn("brown", plan.must_terms)
        self.assertNotIn("color", plan.filters)

    def test_no_preference_is_carried_for_retrieval_to_skip(self) -> None:
        state = self.state_with(operation("no_preference", attribute="material"))
        plan = self.builder.build(state)
        self.assertIn("material", plan.no_preference)


class ProfileIsolationTests(unittest.TestCase):
    """DG-02. The profile is a soft, decaying, separable signal."""

    def setUp(self) -> None:
        self.reducer = StateReducer()

    def build(self, weight: float, profile=PROFILE):
        state = new_session("p", profile)
        self.reducer.apply(
            state, delta(add("category", "boots", strength="hard")), 1
        )
        builder = ActiveQueryBuilder(QueryBuilderConfig(profile_weight=weight))
        return builder.build(state)

    def test_weight_zero_is_indistinguishable_from_having_no_profile(self) -> None:
        with_profile = self.build(0.0, profile=PROFILE)
        without_profile = self.build(0.0, profile={})
        self.assertEqual(with_profile, without_profile)
        self.assertEqual(with_profile.profile_bias.terms, ())
        self.assertEqual(with_profile.profile_bias.weight, 0.0)

    def test_enabled_profile_is_separable_from_dialogue_terms(self) -> None:
        plan = self.build(0.5)
        self.assertIn("fit", plan.profile_bias.terms)
        self.assertNotIn("fit", plan.must_terms)
        self.assertNotIn("fit", plan.should_terms)
        self.assertNotIn("fit", plan.filters)

    def test_profile_never_becomes_a_hard_filter(self) -> None:
        plan = self.build(1.0)
        for value in plan.filters.values():
            self.assertNotIn("comfort", str(value))

    def test_profile_weight_decays_as_constraints_accumulate(self) -> None:
        state = new_session("decay", PROFILE)
        builder = ActiveQueryBuilder(QueryBuilderConfig(profile_weight=1.0))
        self.reducer.apply(state, delta(add("category", "boots")), 1)
        first = builder.build(state).profile_bias.weight
        self.reducer.apply(state, delta(add("color", "brown"), add("size", "10")), 2)
        later = builder.build(state).profile_bias.weight
        self.assertLess(later, first)

    def test_explicit_constraint_outranks_a_contradicting_profile_tag(self) -> None:
        profile = {"preference_tags": ["leather"]}
        state = new_session("conflict", profile)
        self.reducer.apply(
            state,
            delta(
                operation(
                    "exclude",
                    attribute="material",
                    new_value="leather",
                    polarity="exclude",
                    strength="hard",
                    confidence=0.9,
                )
            ),
            1,
        )
        builder = ActiveQueryBuilder(QueryBuilderConfig(profile_weight=1.0))
        plan = builder.build(state)
        self.assertIn("leather", plan.exclude_terms)
        self.assertNotIn("leather", plan.must_terms)
        self.assertNotIn("leather", plan.filters)


class RetrievalHandoffTests(unittest.TestCase):
    """The plan has to survive Person 2's adapter, not just our own asserts."""

    def setUp(self) -> None:
        self.reducer = StateReducer()

    def test_plan_adapts_into_a_retrieval_request(self) -> None:
        state = new_session("handoff", PROFILE)
        self.reducer.apply(
            state,
            delta(
                add("category", "hiking boots", strength="hard"),
                add("budget", "under $120", strength="hard"),
                add("color", "brown"),
            ),
            1,
        )
        builder = ActiveQueryBuilder(QueryBuilderConfig(profile_weight=0.4))
        plan = builder.build(state)
        request = request_from_search_plan(plan)

        self.assertEqual(request.intent_version, 1)
        self.assertEqual(request.mode, "buying")
        attributes = {constraint.attribute for constraint in request.constraints}
        self.assertIn("category", attributes)
        self.assertIn("budget", attributes)
        budget = next(c for c in request.constraints if c.attribute == "budget")
        self.assertEqual(budget.operator, "lte")
        self.assertEqual(tuple(budget.values), (120.0,))
        self.assertIn("fit", request.profile_terms)
        self.assertGreater(request.profile_weight, 0.0)

    def test_no_preference_attribute_is_dropped_by_the_adapter(self) -> None:
        state = new_session("skip", PROFILE)
        self.reducer.apply(
            state,
            delta(
                add("category", "boots", strength="hard"),
                operation("no_preference", attribute="material"),
            ),
            1,
        )
        plan = ActiveQueryBuilder().build(state)
        request = request_from_search_plan(plan)
        attributes = {constraint.attribute for constraint in request.constraints}
        self.assertNotIn("material", attributes)


if __name__ == "__main__":
    unittest.main()
