"""The live probe's safeguards, exercised without spending anything.

A probe that silently degrades is worse than no probe: it produces a clean,
plausible, wrong number that then gets reported as an API result. Everything
here checks the guards rather than the API.
"""

from __future__ import annotations

import pathlib
import unittest

from evaluator.local_evaluator import load_jsonl
from scripts.live_probe import UsageRecordingAgent, stratified
from tikitaka.contracts.domain import Usage
from tikitaka.models.factory import PRIMARY_ROUTE, GatewaySelection
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.orchestration.runtime import (
    DeterministicRuntimeConfig,
    RuntimeConfig,
    build_agent,
    build_deterministic_agent,
)

CATALOG = pathlib.Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"
DATASET = pathlib.Path("data/public_set.jsonl")


class PricedInterpreter:
    """Reports usage the way a real API route would, without a network."""

    def __init__(self) -> None:
        self._inner = HeuristicInterpreter()

    def interpret(self, message: str, state: object):
        delta, _ = self._inner.interpret(message, state)
        return delta, Usage(
            prompt_tokens=400,
            completion_tokens=150,
            reasoning_tokens=80,
            calls=1,
            repairs=1,
            latency_ms=1234.5,
            estimated_cost=0.0026,
            route=PRIMARY_ROUTE.route_id,
        )


class StratifiedSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET.exists():
            raise unittest.SkipTest("public set not present")
        cls.samples = load_jsonl(DATASET)

    def test_every_scenario_survives_a_small_sample(self) -> None:
        # Proportional sampling alone drops Boundary at n=10 (it is 10 of 200),
        # and Boundary is where the no-preference handling lives.
        scenarios = {sample["scenario_type"] for sample in self.samples}
        for count in (4, 10, 20):
            picked = stratified(self.samples, count, seed=2026)
            self.assertEqual(len(picked), count)
            self.assertEqual({item["scenario_type"] for item in picked}, scenarios)

    def test_the_same_seed_picks_the_same_sessions(self) -> None:
        first = [item["sample_id"] for item in stratified(self.samples, 10, 2026)]
        second = [item["sample_id"] for item in stratified(self.samples, 10, 2026)]
        self.assertEqual(first, second)

    def test_a_different_seed_picks_differently(self) -> None:
        first = [item["sample_id"] for item in stratified(self.samples, 10, 2026)]
        other = [item["sample_id"] for item in stratified(self.samples, 10, 7)]
        self.assertNotEqual(first, other)

    def test_zero_sessions_is_empty_not_an_error(self) -> None:
        self.assertEqual(stratified(self.samples, 0, 2026), [])


class UsageHarvestTests(unittest.TestCase):
    def _run(self, agent) -> dict:
        recorder = UsageRecordingAgent(agent)
        with recorder:
            recorder.reset("probe-1", {})
            recorder.respond("probe-1", "I'm looking for shoes.", 1, 10)
            recorder.respond("probe-1", "Cotton, please.", 2, 10)
            return recorder.harvest()

    def test_it_collects_the_fields_the_evaluator_does_not_report(self) -> None:
        # Reasoning tokens, repairs and latency are the whole point of the
        # probe, and the official response payload carries none of them.
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=GatewaySelection(
                interpreter=PricedInterpreter(),
                text_model=None,
                route=PRIMARY_ROUTE,
                degraded=False,
            ),
        )
        harvest = self._run(agent)
        self.assertEqual(harvest["sessions"], 1)
        self.assertEqual(harvest["calls"], 2)
        self.assertEqual(harvest["prompt_tokens"], 800)
        self.assertEqual(harvest["reasoning_tokens"], 160)
        self.assertEqual(harvest["repairs"], 2)
        self.assertGreater(harvest["estimated_cost"], 0.0)
        self.assertGreater(harvest["latency_ms_mean"], 0.0)
        # The reranker also records an event, at zero calls, so attribution
        # is asserted per component rather than by exact dict equality.
        self.assertEqual(harvest["calls_by_component"]["interpreter"], 2)

    def test_a_degraded_run_harvests_zero_tokens(self) -> None:
        # This is the condition the probe treats as fatal. If it were ever to
        # report silently, a deterministic run would be published as an API one.
        harvest = self._run(
            build_deterministic_agent(CATALOG, DeterministicRuntimeConfig())
        )
        self.assertEqual(harvest["prompt_tokens"], 0)
        self.assertEqual(harvest["completion_tokens"], 0)

    def test_fallback_turns_are_counted_separately(self) -> None:
        class Failing:
            def interpret(self, message, state):
                raise RuntimeError("provider down")

        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=GatewaySelection(
                interpreter=Failing(),
                text_model=None,
                route=PRIMARY_ROUTE,
                degraded=False,
            ),
        )
        self.assertEqual(self._run(agent)["fallback_turns"], 2)


class PartialDegradationTests(unittest.TestCase):
    """The dangerous degradation is partial, not total."""

    def test_a_mostly_failing_route_is_detectable_from_the_harvest(self) -> None:
        # Zero tokens is easy to catch. A route that fails most turns still
        # produces tokens, still completes, and still prints a plausible hit
        # rate that actually belongs to the fallback. Observed for real: the
        # account ran out of credits and 60 of 63 escalations returned 429.
        class Failing:
            def interpret(self, message, state):
                raise RuntimeError("HTTP 429: no credits remaining")

        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=GatewaySelection(
                interpreter=Failing(),
                text_model=None,
                route=PRIMARY_ROUTE,
                degraded=False,
            ),
        )
        recorder = UsageRecordingAgent(agent)
        with recorder:
            recorder.reset("p", {})
            for turn in (1, 2, 3):
                recorder.respond("p", "I want shoes.", turn, 10)
            harvest = recorder.harvest()

        attempted = harvest["calls"] + harvest["fallback_turns"]
        self.assertEqual(harvest["fallback_turns"], 3)
        self.assertGreater(harvest["fallback_turns"] / attempted, 0.2)

    def test_a_healthy_route_reports_no_degradation(self) -> None:
        agent, _ = build_agent(
            CATALOG,
            RuntimeConfig(enable_llm_reranker=False),
            environ={},
            model_selection=GatewaySelection(
                interpreter=PricedInterpreter(),
                text_model=None,
                route=PRIMARY_ROUTE,
                degraded=False,
            ),
        )
        recorder = UsageRecordingAgent(agent)
        with recorder:
            recorder.reset("p", {})
            recorder.respond("p", "I want shoes.", 1, 10)
            harvest = recorder.harvest()
        self.assertEqual(harvest["fallback_turns"], 0)


if __name__ == "__main__":
    unittest.main()
