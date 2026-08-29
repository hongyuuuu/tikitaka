from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.retrieval_fakes import GatewaySemanticFakeModel
from tikitaka.contracts import (
    DecisionReasonCode,
    TurnAction,
    TurnDecision,
)
from tikitaka.models.base import ModelRoute
from tikitaka.models.fake import ScriptedInterpreter
from tikitaka.orchestration.sessions import SessionRegistry
from tikitaka.orchestration.shopping_agent import ShoppingAgent
from tikitaka.ranking import DeterministicRanker
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import build_dense_artifact, load_dense_index
from tikitaka.retrieval.embedding import GatewayEmbedder
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.state.query_builder import ActiveQueryBuilder, QueryBuilderConfig
from tikitaka.state.reducer import StateReducer
from tikitaka.state.schema import make_delta, operation
from tikitaka.state.session import SessionState, new_session


CATALOG = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"
ROUTE = ModelRoute(
    route_id="fixture/owner-integration-v1",
    provider="fixture-gateway",
    model="semantic-keywords-v1",
    pinned=True,
)


class AlwaysRecommend:
    def choose(self, state, candidates, turn):
        return TurnDecision(
            action=TurnAction.RECOMMEND,
            ask_attribute=None,
            reason_code=DecisionReasonCode.RANKING_STABLE,
            reason="owner-integration retrieval assertion",
            expected_information_gain=0.0,
        )


class RetrievalOwnerIntegrationTest(unittest.TestCase):
    def test_override_flows_from_owner_state_to_fresh_hybrid_ranking(self) -> None:
        catalog = load_catalog(CATALOG)
        first_intent = make_delta(
            inferred_mode="buying",
            mode_confidence=0.95,
            generality=0.2,
            operations=(
                operation(
                    "add",
                    attribute="category",
                    new_value="shoes",
                    polarity="include",
                    strength="hard",
                    confidence=0.95,
                ),
                operation(
                    "add",
                    attribute="material",
                    new_value="leather",
                    polarity="include",
                    strength="hard",
                    confidence=0.9,
                ),
                operation(
                    "add",
                    attribute="use_case",
                    new_value="hiking",
                    polarity="include",
                    strength="soft",
                    confidence=0.85,
                ),
                operation(
                    "add",
                    attribute="budget",
                    new_value=80,
                    polarity="include",
                    strength="hard",
                    confidence=0.95,
                ),
            ),
        )
        category_override = make_delta(
            inferred_mode="buying",
            mode_confidence=0.95,
            generality=0.2,
            operations=(
                operation(
                    "add",
                    attribute="category",
                    new_value="bag",
                    polarity="include",
                    strength="hard",
                    confidence=0.95,
                ),
                operation(
                    "add",
                    attribute="material",
                    new_value="canvas",
                    polarity="include",
                    strength="hard",
                    confidence=0.9,
                ),
                operation(
                    "add",
                    attribute="use_case",
                    new_value="travel",
                    polarity="include",
                    strength="soft",
                    confidence=0.85,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            build_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), ROUTE)
            manifest = build_dense_artifact(
                catalog,
                build_embedder,
                directory,
                embedding_provider=ROUTE.provider,
                embedding_model=ROUTE.model,
            )
            index = load_dense_index(directory, catalog)
            query_route = ModelRoute(
                route_id=ROUTE.route_id,
                provider=ROUTE.provider,
                model=ROUTE.model,
                index_id=manifest.index_id,
                pinned=True,
            )
            query_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), query_route)
            sessions: SessionRegistry[SessionState] = SessionRegistry(new_session)
            with HybridRetriever(
                catalog,
                dense_index=index,
                query_embedder=query_embedder,
            ) as retriever:
                agent = ShoppingAgent(
                    sessions=sessions,
                    reducer=StateReducer(),
                    interpreter=ScriptedInterpreter((first_intent, category_override)),
                    query_builder=ActiveQueryBuilder(
                        QueryBuilderConfig(
                            route_policy="hybrid",
                            embedding_route_id=manifest.route_id,
                            index_id=manifest.index_id,
                        )
                    ),
                    retriever=retriever,
                    decision_policy=AlwaysRecommend(),
                    reranker=DeterministicRanker(),
                    catalog_ids=catalog.ids,
                    candidate_limit=7,
                )
                agent.reset("isolated-session", {"preference_tags": ["comfort"]})
                first = agent.respond(
                    "isolated-session",
                    "I need leather hiking shoes under $80.",
                    1,
                    10,
                )
                second = agent.respond(
                    "isolated-session",
                    "Actually, make that a canvas travel bag.",
                    2,
                    10,
                )

        first_ids = [item["parent_asin"] for item in first["recommendations"]]
        second_ids = [item["parent_asin"] for item in second["recommendations"]]
        state = sessions.get("isolated-session")
        self.assertEqual(first_ids[0], "A_HIKE")
        self.assertEqual(second_ids[0], "G_TOTE")
        self.assertEqual(state.intent_version, 2)
        self.assertEqual(
            {str(item.attribute): item.normalized_value for item in state.active_constraints},
            {
                "category": "bag",
                "material": "canvas",
                "use_case": "travel",
                "budget": 80.0,
            },
        )
        self.assertEqual(query_embedder.usage.calls, 2)
        self.assertTrue(set(first_ids).issubset(catalog.ids))
        self.assertTrue(set(second_ids).issubset(catalog.ids))


if __name__ == "__main__":
    unittest.main()
