from __future__ import annotations

import re
from typing import Sequence


class SemanticFakeEmbedder:
    """Deterministic fixture embedder; never a production model route."""

    route_id = "fixture-semantic-v1"

    _groups = (
        ("hiking", "trekking", "trail", "mountain", "outdoor"),
        ("waterproof", "water-resistant", "water resistant", "rainproof", "wet", "storm"),
        ("running", "jogging", "gym", "athletic", "fitness"),
        ("walking", "walks", "comfort", "comfortable", "cushioned", "travel"),
        ("bag", "tote", "carryall", "handbag"),
        ("coat", "winter", "warm", "wool"),
        ("formal", "fashion", "dress", "polished"),
    )

    def __init__(self, *, fail_after_document_calls: int | None = None) -> None:
        self.fail_after_document_calls = fail_after_document_calls
        self.document_calls = 0

    @classmethod
    def _embed(cls, text: str) -> tuple[float, ...]:
        normalized = re.sub(r"\s+", " ", text.casefold())
        values = [
            float(sum(normalized.count(term) for term in group))
            for group in cls._groups
        ]
        values.append(0.1)
        return tuple(values)

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if (
            self.fail_after_document_calls is not None
            and self.document_calls >= self.fail_after_document_calls
        ):
            raise RuntimeError("intentional fixture embedding failure")
        self.document_calls += 1
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)


class FailingQueryEmbedder(SemanticFakeEmbedder):
    def embed_query(self, text: str) -> tuple[float, ...]:
        raise TimeoutError("intentional fixture query timeout")
