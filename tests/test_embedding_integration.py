from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from tests.retrieval_fakes import GatewaySemanticFakeModel
from tikitaka.contracts import Embedder, Usage
from tikitaka.models.base import ModelRoute
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import (
    DenseRouteError,
    build_dense_artifact,
    embed_query_for_index,
    load_dense_index,
)
from tikitaka.retrieval.embedding import EmbeddingAdapterError, GatewayEmbedder
from tikitaka.retrieval.hybrid import HybridRetriever
from tikitaka.retrieval.request import RetrievalRequest


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"
ROUTE = ModelRoute(
    route_id="fixture/gateway-semantic-v1",
    provider="fixture-gateway",
    model="semantic-keywords-v1",
    pinned=True,
)


class GatewayEmbeddingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(FIXTURE)

    def test_bridge_satisfies_shared_embedder_and_accumulates_usage(self) -> None:
        model = GatewaySemanticFakeModel()
        embedder = GatewayEmbedder(model, ROUTE)

        self.assertIsInstance(embedder, Embedder)
        documents = embedder.embed_documents(("walking shoes", "hiking boots"))
        query = embedder.embed_query("comfortable travel footwear")

        self.assertEqual(len(documents), 2)
        self.assertEqual(len(query), len(documents[0]))
        self.assertEqual(embedder.usage.calls, 2)
        self.assertEqual(embedder.usage.route, ROUTE.route_id)
        taken = embedder.take_usage()
        self.assertEqual(taken.calls, 2)
        self.assertEqual(embedder.usage.calls, 0)

    def test_bridge_rejects_bad_batch_and_usage_identity(self) -> None:
        wrong_count = GatewayEmbedder(
            GatewaySemanticFakeModel(wrong_count=True),
            ROUTE,
        )
        with self.assertRaisesRegex(EmbeddingAdapterError, "vectors for 1 texts"):
            wrong_count.embed_query("shoes")

        wrong_usage = GatewayEmbedder(
            GatewaySemanticFakeModel(wrong_usage_route=True),
            ROUTE,
        )
        with self.assertRaisesRegex(EmbeddingAdapterError, "identity mismatch: route"):
            wrong_usage.embed_query("shoes")

        class MissingUsageModel:
            def embed(self, texts, route):
                return [[1.0] for _ in texts], Usage()

        missing_usage = GatewayEmbedder(MissingUsageModel(), ROUTE)
        with self.assertRaisesRegex(EmbeddingAdapterError, "report a call"):
            missing_usage.embed_query("shoes")

        class InvalidVectorModel:
            def __init__(self, vectors):
                self.vectors = vectors

            def embed(self, texts, route):
                from tikitaka.models.usage import for_route

                return self.vectors, for_route(route)

        nonfinite = GatewayEmbedder(InvalidVectorModel([[float("nan")]]), ROUTE)
        with self.assertRaisesRegex(EmbeddingAdapterError, "non-finite"):
            nonfinite.embed_query("shoes")

        inconsistent = GatewayEmbedder(
            InvalidVectorModel([[1.0], [1.0, 2.0]]),
            ROUTE,
        )
        with self.assertRaisesRegex(EmbeddingAdapterError, "inconsistent dimensions"):
            inconsistent.embed_documents(("one", "two"))

        valid = GatewayEmbedder(GatewaySemanticFakeModel(), ROUTE)
        with self.assertRaisesRegex(EmbeddingAdapterError, "only strings"):
            valid.embed_documents(("valid", None))

    def test_gateway_identity_is_bound_to_built_and_loaded_index(self) -> None:
        build_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), ROUTE)
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dense_artifact(
                self.catalog,
                build_embedder,
                directory,
                embedding_provider=ROUTE.provider,
                embedding_model=ROUTE.model,
                batch_size=3,
            )
            index = load_dense_index(directory, self.catalog)
            query_route = ModelRoute(
                route_id=ROUTE.route_id,
                provider=ROUTE.provider,
                model=ROUTE.model,
                index_id=manifest.index_id,
                pinned=True,
            )
            query_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), query_route)
            vector = embed_query_for_index(query_embedder, index, "rainproof mountain footwear")
            hits = index.search(vector, limit=3)
            with HybridRetriever(
                self.catalog,
                dense_index=index,
                query_embedder=query_embedder,
            ):
                pass

        self.assertTrue(hits)
        self.assertIn("A_HIKE", [hit.parent_asin for hit in hits])

    def test_gateway_timeout_degrades_hybrid_query_to_sparse(self) -> None:
        build_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), ROUTE)
        with tempfile.TemporaryDirectory() as directory:
            build_dense_artifact(
                self.catalog,
                build_embedder,
                directory,
                embedding_provider=ROUTE.provider,
                embedding_model=ROUTE.model,
            )
            index = load_dense_index(directory, self.catalog)
            failing = GatewayEmbedder(GatewaySemanticFakeModel(fail=True), ROUTE)
            with HybridRetriever(
                self.catalog,
                dense_index=index,
                query_embedder=failing,
            ) as retriever:
                result = retriever.retrieve(
                    RetrievalRequest(
                        text_query="waterproof hiking boots",
                        route_policy="hybrid",
                        embedding_route_id=index.manifest.route_id,
                        index_id=index.manifest.index_id,
                    ),
                    limit=5,
                )

        self.assertEqual(result.diagnostics.executed_route, "sparse_fallback")
        self.assertIn("dense_query_failed", result.diagnostics.route_failures)
        self.assertTrue(result.hits)

    def test_provider_model_and_index_mismatches_fail_closed(self) -> None:
        build_embedder = GatewayEmbedder(GatewaySemanticFakeModel(), ROUTE)
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dense_artifact(
                self.catalog,
                build_embedder,
                directory,
                embedding_provider=ROUTE.provider,
                embedding_model=ROUTE.model,
            )
            index = load_dense_index(directory, self.catalog)
            for field, value in (
                ("provider", "wrong-provider"),
                ("model", "wrong-model"),
                ("index_id", "dense-wrong-index-id"),
            ):
                values = {
                    "route_id": ROUTE.route_id,
                    "provider": ROUTE.provider,
                    "model": ROUTE.model,
                    "index_id": manifest.index_id,
                }
                values[field] = value
                embedder = GatewayEmbedder(
                    GatewaySemanticFakeModel(),
                    ModelRoute(**values),
                )
                with self.subTest(field=field):
                    with self.assertRaisesRegex(DenseRouteError, field):
                        embed_query_for_index(embedder, index, "walking shoes")

    def test_wrong_declared_index_stops_build_before_artifact_progress(self) -> None:
        route = ModelRoute(
            route_id=ROUTE.route_id,
            provider=ROUTE.provider,
            model=ROUTE.model,
            index_id="dense-wrong-index-id",
        )
        embedder = GatewayEmbedder(GatewaySemanticFakeModel(), route)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DenseRouteError, "artifact being built"):
                build_dense_artifact(
                    self.catalog,
                    embedder,
                    directory,
                    embedding_provider=ROUTE.provider,
                    embedding_model=ROUTE.model,
                    batch_size=3,
                )
            checkpoint = json.loads(
                (Path(directory) / "build.checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["next_index"], 0)
            self.assertEqual((Path(directory) / "ids.jsonl.partial").stat().st_size, 0)
            self.assertEqual((Path(directory) / "vectors.f32.partial").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
