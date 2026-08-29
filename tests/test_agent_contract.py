from __future__ import annotations

import unittest

from starter.agent import Agent


class DelegateSpy:
    def __init__(self) -> None:
        self.calls = []

    def reset(self, session_id, user_profile):
        self.calls.append(("reset", session_id, user_profile))

    def respond(self, session_id, user_message, turn, top_k):
        self.calls.append(("respond", session_id, user_message, turn, top_k))
        return {
            "message": "delegated",
            "ask_attribute": None,
            "recommendations": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class AgentContractTest(unittest.TestCase):
    def test_official_agent_is_a_thin_delegate(self) -> None:
        spy = DelegateSpy()
        agent = Agent(shopping_agent=spy)
        profile = {"summary": "visible"}
        agent.reset("session", profile)
        response = agent.respond("session", "message", 1, 10)

        self.assertEqual(spy.calls, [
            ("reset", "session", profile),
            ("respond", "session", "message", 1, 10),
        ])
        self.assertEqual(response["message"], "delegated")

    def test_default_adapter_emits_official_shape_on_tiny_catalog(self) -> None:
        agent = Agent("tests/fixtures/tiny_catalog.jsonl")
        self.addCleanup(agent.close)
        agent.reset("session", {
            "purchase_frequency": "monthly",
            "average_prior_rating": 4.0,
            "rating_style": "balanced",
            "preference_tags": [],
            "summary": "",
        })
        response = agent.respond("session", "blue cotton walking shoe", 1, 10)
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "TINY-A")
        self.assertEqual(set(response["usage"]), {"prompt_tokens", "completion_tokens"})


if __name__ == "__main__":
    unittest.main()
