"""Official competition adapter; runtime behavior lives under ``tikitaka``."""

from __future__ import annotations

from pathlib import Path

from tikitaka.orchestration.runtime import build_deterministic_agent


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", *, shopping_agent=None) -> None:
        self._shopping_agent = shopping_agent or build_deterministic_agent(catalog_path)

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
