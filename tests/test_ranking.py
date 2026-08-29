from __future__ import annotations

import unittest
import json
from dataclasses import dataclass, field
from pathlib import Path

from tikitaka.ranking.constraints import assess_candidate
from tikitaka.ranking.deterministic import DeterministicRanker, DeterministicRankerConfig
from tikitaka.ranking.diversity import DiversityConfig, diversify
from tikitaka.ranking.llm import LLMReranker, LLMRerankerConfig


@dataclass(frozen=True)
class FakeConstraint:
    attribute: str
    value: object
    normalized_value: object
    polarity: str = "include"
    strength: str = "hard"
    source_turn: int = 1
    confidence: float = 1.0
    intent_version: int = 1
    status: str = "active"
    category_dependent: bool = False


@dataclass(frozen=True)
class FakeEvidence:
    matched_fields: tuple[str, ...] = ()
    supporting_snippets: tuple[str, ...] = ()
    constraint_outcomes: dict[str, str] = field(default_factory=dict)
    attribute_values: dict[str, tuple[object, ...]] = field(default_factory=dict)
    evidence_reliability: dict[str, float] = field(default_factory=dict)
    unknown_fields: tuple[str, ...] = ()
    route_details: dict[str, object] = field(default_factory=dict)
    profile_contribution: float = 0.0


@dataclass(frozen=True)
class FakeCandidate:
    parent_asin: str
    product_evidence: FakeEvidence
    sparse_rank: int | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    structural_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class FakeState:
    session_id: str = "session"
    turn: int = 1
    mode: str = "buying"
    mode_confidence: float = 0.9
    intent_version: int = 1
    active_constraints: tuple[FakeConstraint, ...] = ()
    revalidation_constraints: tuple[FakeConstraint, ...] = ()
    no_preference: frozenset[str] = frozenset()
    asked_attributes: frozenset[str] = frozenset()
    shown_product_ids: frozenset[str] = frozenset()
    profile_seed: dict[str, object] = field(default_factory=dict)


def candidate(
    parent_asin: str,
    fused: float,
    *,
    outcomes: dict[str, str] | None = None,
    reliability: dict[str, float] | None = None,
    values: dict[str, tuple[object, ...]] | None = None,
    profile: float = 0.0,
    sparse_rank: int | None = None,
    dense_rank: int | None = None,
) -> FakeCandidate:
    return FakeCandidate(
        parent_asin=parent_asin,
        product_evidence=FakeEvidence(
            constraint_outcomes=outcomes or {},
            evidence_reliability=reliability or {},
            attribute_values=values or {},
            profile_contribution=profile,
            supporting_snippets=(f"evidence for {parent_asin}",),
        ),
        sparse_rank=sparse_rank,
        dense_rank=dense_rank,
        structural_score=fused / 2.0,
        fused_score=fused,
    )


class DeterministicRankingTests(unittest.TestCase):
    def test_synthetic_fixture_contains_unique_catalog_ids(self) -> None:
        path = Path(__file__).parent / "fixtures" / "decision_catalog.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = [row["parent_asin"] for row in rows]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(ids), len(set(ids)))

    def test_reliable_hard_contradiction_is_excluded(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        contradicted = candidate(
            "BAD",
            1.0,
            outcomes={"material": "contradiction"},
            reliability={"material": 0.95},
        )
        matched = candidate(
            "GOOD",
            0.3,
            outcomes={"material": "match"},
            reliability={"material": 0.95},
        )
        ids, _ = DeterministicRanker().rank(state, [contradicted, matched], 10)
        self.assertEqual(ids, ["GOOD"])

    def test_unknown_metadata_is_not_a_contradiction(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        unknown = candidate(
            "UNKNOWN",
            0.8,
            outcomes={"material": "unknown"},
            reliability={"material": 0.0},
        )
        assessment = assess_candidate(state, unknown)
        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.unknown_count, 1)
        self.assertEqual(DeterministicRanker().rank(state, [unknown], 10)[0], ["UNKNOWN"])

    def test_all_confirmed_hard_contradictions_remain_excluded(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        contradicted = [
            candidate(
                parent_asin,
                fused,
                outcomes={"material": "contradiction"},
                reliability={"material": 1.0},
            )
            for parent_asin, fused in (("A", 1.0), ("B", 0.8))
        ]
        self.assertEqual(DeterministicRanker().rank(state, contradicted, 10)[0], [])

    def test_unreliable_contradiction_is_soft_not_excluded(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("color", "red", "red"),)
        )
        weak = candidate(
            "WEAK",
            1.0,
            outcomes={"color": "contradiction"},
            reliability={"color": 0.4},
        )
        assessment = assess_candidate(state, weak)
        self.assertTrue(assessment.eligible)
        self.assertGreater(assessment.soft_contradiction_score, 0.0)

    def test_low_confidence_hard_constraint_does_not_filter(self) -> None:
        state = FakeState(
            active_constraints=(
                FakeConstraint("color", "red", "red", confidence=0.4),
            )
        )
        contradicted = candidate(
            "CANDIDATE",
            1.0,
            outcomes={"color": "contradiction"},
            reliability={"color": 1.0},
        )
        assessment = assess_candidate(state, contradicted)
        self.assertTrue(assessment.eligible)
        self.assertGreater(assessment.soft_contradiction_score, 0.0)

    def test_constraint_match_can_improve_order(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        plain = candidate("PLAIN", 0.8, outcomes={"material": "unknown"})
        match = candidate(
            "MATCH",
            0.7,
            outcomes={"material": "match"},
            reliability={"material": 1.0},
        )
        ranker = DeterministicRanker(
            DeterministicRankerConfig(
                fused_weight=0.10,
                structural_weight=0.05,
                route_agreement_weight=0.0,
                constraint_match_weight=0.80,
            )
        )
        ids, _ = ranker.rank(state, [plain, match], 10)
        self.assertEqual(ids[0], "MATCH")

    def test_duplicate_and_empty_ids_are_removed(self) -> None:
        state = FakeState()
        candidates = [candidate("A", 0.8), candidate("A", 0.7), candidate("", 1.0)]
        self.assertEqual(DeterministicRanker().rank(state, candidates, 10)[0], ["A"])

    def test_stable_tie_break_ends_in_parent_asin(self) -> None:
        state = FakeState()
        ids, _ = DeterministicRanker().rank(
            state, [candidate("B", 0.5), candidate("A", 0.5)], 10
        )
        self.assertEqual(ids, ["A", "B"])

    def test_same_intent_shown_products_are_backfill_after_unseen(self) -> None:
        state = FakeState(shown_product_ids=frozenset({"SHOWN"}))
        ids, _ = DeterministicRanker().rank(
            state, [candidate("SHOWN", 1.0), candidate("NEW", 0.5)], 10
        )
        self.assertEqual(ids, ["NEW", "SHOWN"])

    def test_shown_product_remains_fallback_when_everything_was_shown(self) -> None:
        state = FakeState(shown_product_ids=frozenset({"A", "B"}))
        ids, _ = DeterministicRanker().rank(
            state, [candidate("A", 1.0), candidate("B", 0.5)], 10
        )
        self.assertEqual(set(ids), {"A", "B"})

    def test_new_intent_view_makes_old_products_eligible(self) -> None:
        old_state = FakeState(intent_version=1, shown_product_ids=frozenset({"A"}))
        new_state = FakeState(intent_version=2, shown_product_ids=frozenset())
        candidates = [candidate("A", 1.0), candidate("B", 0.5)]
        self.assertEqual(DeterministicRanker().rank(old_state, candidates, 10)[0][0], "B")
        self.assertEqual(DeterministicRanker().rank(new_state, candidates, 10)[0][0], "A")

    def test_shown_products_do_not_leave_top_k_slots_empty(self) -> None:
        state = FakeState(shown_product_ids=frozenset({"S1", "S2"}))
        ids, _ = DeterministicRanker().rank(
            state,
            [candidate("S1", 1.0), candidate("S2", 0.9), candidate("NEW", 0.2)],
            3,
        )
        self.assertEqual(ids, ["NEW", "S1", "S2"])

    def test_profile_signal_defaults_to_zero_weight(self) -> None:
        state = FakeState()
        profiled = candidate("PROFILE", 0.4, profile=1.0)
        stronger = candidate("STRONGER", 0.6, profile=0.0)
        self.assertEqual(
            DeterministicRanker().rank(state, [profiled, stronger], 10)[0][0],
            "STRONGER",
        )

    def test_top_k_is_respected_and_usage_is_zero(self) -> None:
        ids, usage = DeterministicRanker().rank(
            FakeState(), [candidate(str(index), 1.0 / index) for index in range(1, 6)], 3
        )
        self.assertEqual(len(ids), 3)
        self.assertEqual(usage.calls, 0)
        self.assertEqual(usage.route, "deterministic")


class DiversityTests(unittest.TestCase):
    def test_diversity_can_select_a_different_unconstrained_category(self) -> None:
        ranker = DeterministicRanker()
        state = FakeState(mode="browsing")
        ranked = ranker.rank_candidates(
            state,
            [
                candidate("A", 1.0, values={"category": ("shoes",)}),
                candidate("B", 0.95, values={"category": ("shoes",)}),
                candidate("C", 0.90, values={"category": ("boots",)}),
            ],
        )
        selected = diversify(state, ranked, 2, DiversityConfig(strength=0.45))
        self.assertEqual([item.parent_asin for item in selected], ["A", "C"])

    def test_explicit_category_disables_category_diversification(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("category", "shoes", "shoes"),)
        )
        ranked = DeterministicRanker().rank_candidates(
            state,
            [
                candidate("A", 1.0, values={"category": ("shoes",)}),
                candidate("B", 0.95, values={"category": ("shoes",)}),
                candidate("C", 0.90, values={"category": ("boots",)}),
            ],
        )
        selected = diversify(state, ranked, 2, DiversityConfig(strength=0.45))
        self.assertEqual([item.parent_asin for item in selected], ["A", "B"])


class FakeModel:
    def __init__(self, output: object = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.requests = []

    def rerank(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.output, type(
            "Usage",
            (),
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "reasoning_tokens": 2,
                "calls": 1,
                "provider": "fake",
            },
        )()


class LLMRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = FakeState()
        self.candidates = [candidate("A", 1.0), candidate("B", 0.8), candidate("C", 0.6)]

    def test_valid_output_reorders_shortlist(self) -> None:
        model = FakeModel(["C", "A", "B"])
        ids, usage = LLMReranker(model).rank(self.state, self.candidates, 3)
        self.assertEqual(ids, ["C", "A", "B"])
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.model, "gpt-5.6-terra")
        self.assertEqual(model.requests[0].reasoning_level, "xhigh")

    def test_hallucinations_duplicates_and_omissions_are_normalized(self) -> None:
        model = FakeModel({"parent_asins": ["B", "OUTSIDE", "B"]})
        ids, _ = LLMReranker(model).rank(self.state, self.candidates, 3)
        self.assertEqual(ids, ["B", "A", "C"])

    def test_malformed_output_falls_back(self) -> None:
        ids, usage = LLMReranker(FakeModel("not json")).rank(
            self.state, self.candidates, 3
        )
        self.assertEqual(ids, ["A", "B", "C"])
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.route, "deterministic_fallback")

    def test_model_exception_falls_back(self) -> None:
        ids, usage = LLMReranker(FakeModel(error=TimeoutError())).rank(
            self.state, self.candidates, 3
        )
        self.assertEqual(ids, ["A", "B", "C"])
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.route, "deterministic_fallback")

    def test_hard_contradiction_never_reaches_model(self) -> None:
        state = FakeState(
            active_constraints=(FakeConstraint("material", "canvas", "canvas"),)
        )
        bad = candidate(
            "BAD",
            1.0,
            outcomes={"material": "contradiction"},
            reliability={"material": 1.0},
        )
        good = candidate(
            "GOOD",
            0.5,
            outcomes={"material": "match"},
            reliability={"material": 1.0},
        )
        model = FakeModel(["BAD", "GOOD"])
        ids, _ = LLMReranker(model).rank(state, [bad, good], 3)
        self.assertEqual(ids, ["GOOD"])
        self.assertEqual(
            [item["parent_asin"] for item in model.requests[0].candidates], ["GOOD"]
        )

    def test_unapproved_model_route_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LLMRerankerConfig(model="another-model")

    def test_cache_hit_adds_no_current_run_usage(self) -> None:
        class CachedModel:
            def rerank(self, request):
                return ["B", "A"], type(
                    "Usage",
                    (),
                    {
                        "cache_hit": True,
                        "calls": 5,
                        "prompt_tokens": 99,
                        "completion_tokens": 88,
                        "latency_ms": 50.0,
                    },
                )()

        _, usage = LLMReranker(CachedModel()).rank(self.state, self.candidates, 2)
        self.assertTrue(usage.cache_hit)
        self.assertEqual(usage.calls, 0)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
