from __future__ import annotations

import unittest

from tikitaka.config import (
    CONTRACT_VERSION,
    ExperimentPins,
    RuntimeRoutingConfig,
    STRUCTURED_OUTPUT_SCHEMA_VERSION,
)
from tikitaka.contracts import (
    Attribute,
    Candidate,
    Constraint,
    DecisionPolicy,
    DecisionReasonCode,
    EvidenceOutcome,
    IndexManifest,
    IntentInterpreter,
    ProductEvidence,
    ProfileBias,
    QueryBuilder,
    Reranker,
    Retriever,
    RoutePolicy,
    SearchPlan,
    SessionStateView,
    StateDelta,
    StateOperation,
    TurnDecision,
    Usage,
    clamp_unit_interval,
)
from tests.fakes.components import (
    DeterministicReranker,
    DeterministicRetriever,
    FakeQueryBuilder,
    FakeSessionState,
    MalformedInterpreter,
    MalformedReranker,
    RaisingDecisionPolicy,
    RaisingInterpreter,
    RaisingReranker,
    RaisingRetriever,
    ScriptedDecisionPolicy,
    ScriptedInterpreter,
    candidate,
)


class ContractConstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.constraint = Constraint(
            attribute="material",
            value="Cotton",
            normalized_value="cotton",
            polarity="include",
            strength="hard",
            source_turn=1,
            confidence=0.9,
            intent_version=1,
        )
        self.operation = StateOperation(
            operation="add",
            attribute="material",
            old_value=None,
            new_value="cotton",
            scope="attribute",
            polarity="include",
            strength="hard",
            confidence=0.9,
        )

    def test_every_shared_record_constructs_and_normalizes_closed_values(self) -> None:
        delta = StateDelta("buying", 0.8, (self.operation,), 0.2, 0, STRUCTURED_OUTPUT_SCHEMA_VERSION)
        profile = ProfileBias(("walking",), 0.25)
        plan = SearchPlan(
            "cotton shoe", ("shoe",), ("cotton",), (), {"price_max": 80},
            {"material": ("cotton",)}, "buying", 1, frozenset(), frozenset(),
            profile, "hybrid", "embed-v1", "index-v1",
        )
        evidence = ProductEvidence(
            ("title",), ("cotton walking shoe",), {"material": "match"},
            {"material": ("cotton",)}, {"material": 0.9}, ("size",),
            {"sparse": {"rank": 1}}, 0.1,
        )
        product = Candidate("A", evidence, 1, 4.0, 2, 0.8, 0.5, 0.7)
        decision = TurnDecision("recommend", None, "ranking_stable", "stable", 0.1)
        usage = Usage(10, 5, 2, 1, 0, 12.5, "fake", "model", "none", 0.001)

        self.assertEqual(self.constraint.attribute, Attribute.MATERIAL)
        self.assertEqual(delta.inferred_mode.value, "buying")
        self.assertEqual(plan.attribute_values[Attribute.MATERIAL], ("cotton",))
        self.assertEqual(evidence.constraint_outcomes[Attribute.MATERIAL], EvidenceOutcome.MATCH)
        self.assertEqual(product.parent_asin, "A")
        self.assertEqual(decision.reason_code, DecisionReasonCode.RANKING_STABLE)
        self.assertEqual(usage.cost_currency, "USD")
        self.assertEqual(CONTRACT_VERSION, "0.1.0")

    def test_unknown_closed_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Constraint("unknown", "x", "x", "include", "hard", 1, 1.0, 1)
        with self.assertRaises(ValueError):
            StateOperation("invent", None, None, None, "intent", None, None, None)
        with self.assertRaises(ValueError):
            StateDelta("shopping", 1.0, (), 0.0, 0, "v")
        with self.assertRaises(ValueError):
            TurnDecision("wait", None, "ranking_stable", "x", 0.0)
        with self.assertRaises(ValueError):
            ProductEvidence((), (), {"unknown": "match"}, {}, {}, (), {})

    def test_confidence_boundary_behavior_is_explicit(self) -> None:
        self.assertEqual(clamp_unit_interval(-4), 0.0)
        self.assertEqual(clamp_unit_interval(4), 1.0)
        self.assertEqual(clamp_unit_interval(0.4), 0.4)
        with self.assertRaises(ValueError):
            Constraint("material", "x", "x", "include", "hard", 1, 1.01, 1)
        with self.assertRaises(ValueError):
            StateDelta("unknown", -0.01, (), 0.0, 0, "v")
        with self.assertRaises(ValueError):
            ProductEvidence((), (), {}, {}, {"color": 1.1}, (), {})
        with self.assertRaises(ValueError):
            TurnDecision("recommend", None, "ranking_stable", "x", -0.1)

    def test_state_operation_strict_field_rules(self) -> None:
        excluded = StateOperation("exclude", "material", None, "leather", "attribute", "include", "hard", 0.8)
        self.assertEqual(excluded.polarity.value, "exclude")
        with self.assertRaises(ValueError):
            StateOperation("reset", None, None, None, "attribute", None, None, None)
        with self.assertRaises(ValueError):
            StateOperation("remove", None, None, None, "attribute", None, None, None)
        with self.assertRaises(ValueError):
            StateOperation("no_preference", "color", None, "blue", "attribute", None, None, None)

    def test_turn_decision_enforces_mutually_exclusive_action(self) -> None:
        TurnDecision("clarify", "budget", "valuable_clarification", "ask", 0.7)
        with self.assertRaises(ValueError):
            TurnDecision("clarify", None, "valuable_clarification", "ask", 0.7)
        with self.assertRaises(ValueError):
            TurnDecision("recommend", "budget", "ranking_stable", "rank", 0.0)

    def test_usage_cannot_be_negative_or_internally_inconsistent(self) -> None:
        for kwargs in (
            {"prompt_tokens": -1},
            {"latency_ms": -0.1},
            {"estimated_cost": -0.1},
            {"calls": 0, "repairs": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Usage(**kwargs)
        with self.assertRaises(ValueError):
            Usage(cache_hit=True, calls=1)
        self.assertTrue(Usage(cache_hit=True).cache_hit)

    def test_missing_metadata_is_unknown_and_evidence_is_defensively_copied(self) -> None:
        outcomes = {"material": "unknown"}
        evidence = ProductEvidence((), (), outcomes, {}, {"material": 0.0}, ("material",), {})
        outcomes["material"] = "contradiction"
        self.assertEqual(evidence.constraint_outcomes[Attribute.MATERIAL], EvidenceOutcome.UNKNOWN)
        self.assertIn("material", evidence.unknown_fields)

    def test_candidate_preserves_evidence_and_route_identity(self) -> None:
        item = candidate("A", 1, 0.9)
        self.assertEqual(item.sparse_rank, 1)
        self.assertIsNotNone(item.product_evidence)
        with self.assertRaises(ValueError):
            Candidate("A", item.product_evidence, 0, None, None, None, 0.0, 0.0)


class RoutingConfigTest(unittest.TestCase):
    def manifest(self) -> IndexManifest:
        return IndexManifest(
            index_id="index-v1", catalog_checksum="abc", catalog_row_count=3,
            ordered_id_checksum="def", product_text_schema_version="1",
            provider="fake", model="embed", route_id="embed-v1", dimension=4,
            vector_dtype="float32", normalized=True, document_count=3,
            artifact_format="npy", built_at="2026-08-29T00:00:00Z",
            artifact_checksums={"vectors.npy": "123"},
        )

    def test_runtime_routing_and_experiment_pins_are_separate(self) -> None:
        runtime = RuntimeRoutingConfig(embedding_route_id="embed-v1", index_id="index-v1")
        pins = ExperimentPins(retrieval_policy="sparse")
        runtime.validate_index(self.manifest())
        self.assertEqual(runtime.retrieval_policy, RoutePolicy.AUTO)
        self.assertEqual(pins.retrieval_policy, RoutePolicy.SPARSE)

    def test_embedding_identity_is_paired_and_manifest_checked(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeRoutingConfig(embedding_route_id="embed-v1")
        with self.assertRaises(ValueError):
            SearchPlan("", (), (), (), {}, {}, "unknown", 1, frozenset(), frozenset(), ProfileBias(), "dense")
        with self.assertRaises(ValueError):
            RuntimeRoutingConfig(embedding_route_id="wrong", index_id="index-v1").validate_index(self.manifest())


class FakeComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = FakeSessionState()
        self.items = [candidate("B", 2, 0.6), candidate("A", 1, 0.9)]
        self.plan = FakeQueryBuilder().build(self.state)

    def test_fakes_satisfy_published_protocols(self) -> None:
        interpreter = ScriptedInterpreter()
        retriever = DeterministicRetriever(self.items, {"A", "B"})
        decision = ScriptedDecisionPolicy()
        reranker = DeterministicReranker()
        self.assertIsInstance(self.state, SessionStateView)
        self.assertIsInstance(interpreter, IntentInterpreter)
        self.assertIsInstance(FakeQueryBuilder(), QueryBuilder)
        self.assertIsInstance(retriever, Retriever)
        self.assertIsInstance(decision, DecisionPolicy)
        self.assertIsInstance(reranker, Reranker)

    def test_fake_outputs_are_stable_for_fixed_seed_and_configuration(self) -> None:
        first = DeterministicRetriever(self.items, {"A", "B"}, seed=42).search(self.plan, 10)
        second = DeterministicRetriever(self.items, {"A", "B"}, seed=42).search(self.plan, 10)
        self.assertEqual(first, second)
        self.assertEqual([item.parent_asin for item in first], ["A", "B"])

    def test_normal_reranker_cannot_introduce_out_of_shortlist_id(self) -> None:
        ranked, usage = DeterministicReranker(["OUTSIDE", "B"], seed=1).rank(self.state, self.items, 10)
        self.assertEqual(ranked, ["B", "A"])
        self.assertLessEqual(set(ranked), {"A", "B"})
        self.assertEqual(usage.provider, "fake")

    def test_retriever_rejects_non_catalog_candidates(self) -> None:
        with self.assertRaises(ValueError):
            DeterministicRetriever(self.items, {"A"})

    def test_malformed_and_exception_fakes_cover_failure_boundaries(self) -> None:
        malformed_delta, _ = MalformedInterpreter().interpret("x", self.state)
        malformed_ids, _ = MalformedReranker().rank(self.state, self.items, 10)
        self.assertNotIsInstance(malformed_delta, StateDelta)
        self.assertNotEqual(set(malformed_ids) & {"A", "B"}, set(malformed_ids))

        calls = (
            lambda: RaisingInterpreter().interpret("x", self.state),
            lambda: RaisingRetriever().search(self.plan, 10),
            lambda: RaisingDecisionPolicy().choose(self.state, self.items, 1),
            lambda: RaisingReranker().rank(self.state, self.items, 10),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(RuntimeError):
                call()


if __name__ == "__main__":
    unittest.main()
