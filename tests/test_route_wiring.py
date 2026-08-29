"""The official entry point must select the generative route it reports."""

from __future__ import annotations

import pathlib
import unittest

from tikitaka.models.api_llm import ApiInterpreter
from tikitaka.models.base import CredentialMissing
from tikitaka.models.fake import HeuristicInterpreter
from tikitaka.orchestration.runtime import build_agent, build_deterministic_agent

CATALOG = pathlib.Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"
SECRET = "sk-not-a-real-key-0123456789"


def inner(agent) -> object:
    """The interpreter actually reached at runtime."""

    return agent._interpreter._interpreter


class RouteWiringTests(unittest.TestCase):
    def test_no_credential_builds_the_deterministic_route(self) -> None:
        agent, route_id = build_agent(CATALOG, environ={})
        self.assertIsInstance(inner(agent), HeuristicInterpreter)
        self.assertEqual(route_id, "heuristic/local")

    def test_credential_builds_the_api_route(self) -> None:
        agent, route_id = build_agent(CATALOG, environ={"OPENAI_API_KEY": SECRET})
        self.assertIsInstance(inner(agent), ApiInterpreter)
        self.assertEqual(route_id, "primary/gpt-5.6-terra")

    def test_scored_runs_can_refuse_to_degrade(self) -> None:
        with self.assertRaises(CredentialMissing):
            build_agent(CATALOG, environ={}, allow_degraded=False)

    def test_deterministic_builder_is_unchanged_by_default(self) -> None:
        agent = build_deterministic_agent(CATALOG)
        self.assertIsInstance(inner(agent), HeuristicInterpreter)

    def test_reported_route_matches_the_interpreter_in_use(self) -> None:
        """A report must never claim the API while the heuristic did the work."""

        for environ, expected, kind in (
            ({}, "heuristic/local", HeuristicInterpreter),
            ({"OPENAI_API_KEY": SECRET}, "primary/gpt-5.6-terra", ApiInterpreter),
        ):
            with self.subTest(route=expected):
                agent, route_id = build_agent(CATALOG, environ=environ)
                self.assertEqual(route_id, expected)
                self.assertIsInstance(inner(agent), kind)

    def test_credential_does_not_leak_through_the_built_agent(self) -> None:
        agent, _ = build_agent(CATALOG, environ={"OPENAI_API_KEY": SECRET})
        self.assertNotIn(SECRET, repr(inner(agent)))
        self.assertNotIn(SECRET, str(vars(inner(agent))))


if __name__ == "__main__":
    unittest.main()
