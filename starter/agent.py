"""Official competition adapter; runtime behavior lives under ``tikitaka``."""

from __future__ import annotations

from pathlib import Path

from tikitaka.orchestration.runtime import build_submission_agent


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", *, shopping_agent=None) -> None:
        if shopping_agent is None:
            # Uses the API route when a credential is present and the
            # deterministic route otherwise, so the official entry point
            # stays valid with no network.
            shopping_agent, self.route_id = build_submission_agent(catalog_path)
        else:
            self.route_id = "injected"
        self._shopping_agent = shopping_agent
        self.retrieval_route_id = getattr(
            shopping_agent,
            "retrieval_route_id",
            "injected",
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._shopping_agent.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return self._shopping_agent.respond(session_id, user_message, turn, top_k)

    def close(self) -> None:
        close = getattr(self._shopping_agent, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
