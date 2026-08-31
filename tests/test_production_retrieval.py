from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tikitaka.orchestration.production_retrieval import (
    DENSE_ARTIFACT_ENV,
    PRODUCTION_HYBRID_CONFIG,
    ensure_trusted_ca,
    select_production_retrieval,
)


class ProductionRetrievalTests(unittest.TestCase):
    def test_selected_weights_are_frozen_to_validated_hybrid_arm(self) -> None:
        self.assertEqual(PRODUCTION_HYBRID_CONFIG.sparse_weight, 1.0)
        self.assertEqual(PRODUCTION_HYBRID_CONFIG.dense_weight, 0.5)

    def test_missing_artifact_preserves_sparse_fallback(self) -> None:
        selection = select_production_retrieval(
            "catalog.jsonl",
            profile_weight=0.0,
            environ={},
        )

        self.assertIsNone(selection.retriever)
        self.assertEqual(selection.route_id, "sparse")
        self.assertEqual(selection.failure_code, "dense_artifact_unconfigured")
        self.assertEqual(selection.query_builder.route_policy, "sparse")

    def test_valid_artifact_builds_weighted_hybrid_route(self) -> None:
        manifest = SimpleNamespace(route_id="embedding-route", index_id="index-id")
        dense = SimpleNamespace(manifest=manifest)
        embedder = SimpleNamespace(route_id="embedding-route")
        retriever = object()
        environ = {DENSE_ARTIFACT_ENV: "/external/dense", "OPENAI_API_KEY": "secret"}

        with (
            patch(
                "tikitaka.orchestration.production_retrieval.load_catalog",
                return_value=object(),
            ),
            patch(
                "tikitaka.orchestration.production_retrieval.openai_embedder_from_env",
                return_value=embedder,
            ),
            patch(
                "tikitaka.orchestration.production_retrieval.load_dense_index",
                return_value=dense,
            ) as load_index,
            patch(
                "tikitaka.orchestration.production_retrieval.HybridRetriever",
                return_value=retriever,
            ) as build_hybrid,
        ):
            selection = select_production_retrieval(
                "catalog.jsonl",
                profile_weight=0.0,
                environ=environ,
            )

        self.assertIs(selection.retriever, retriever)
        self.assertEqual(selection.route_id, "hybrid/sparse-1/dense-0.5")
        self.assertIsNone(selection.failure_code)
        self.assertEqual(selection.query_builder.route_policy, "hybrid")
        self.assertEqual(selection.query_builder.embedding_route_id, "embedding-route")
        self.assertEqual(selection.query_builder.index_id, "index-id")
        load_index.assert_called_once_with(
            "/external/dense",
            unittest.mock.ANY,
            embedding_route_id="embedding-route",
        )
        self.assertEqual(
            build_hybrid.call_args.kwargs["config"],
            PRODUCTION_HYBRID_CONFIG,
        )

    def test_invalid_artifact_fails_closed_to_sparse(self) -> None:
        environ = {DENSE_ARTIFACT_ENV: "/external/bad", "OPENAI_API_KEY": "secret"}
        with patch(
            "tikitaka.orchestration.production_retrieval.load_catalog",
            side_effect=ValueError("invalid artifact"),
        ):
            selection = select_production_retrieval(
                "catalog.jsonl",
                profile_weight=0.0,
                environ=environ,
            )

        self.assertIsNone(selection.retriever)
        self.assertEqual(selection.route_id, "sparse_fallback")
        self.assertEqual(selection.failure_code, "dense_runtime_unavailable")
        self.assertEqual(selection.query_builder.route_policy, "sparse")

    def test_ca_discovery_preserves_explicit_configuration(self) -> None:
        environ = {"SSL_CERT_FILE": "/custom/ca.pem"}

        self.assertEqual(ensure_trusted_ca(environ), "/custom/ca.pem")
        self.assertEqual(environ["SSL_CERT_FILE"], "/custom/ca.pem")

    def test_ca_discovery_uses_readable_bundle_without_disabling_tls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "ca.pem"
            bundle.write_text("test bundle", encoding="utf-8")
            environ: dict[str, str] = {}
            with patch(
                "tikitaka.orchestration.production_retrieval._ca_bundle_candidates",
                return_value=(bundle,),
            ):
                selected = ensure_trusted_ca(environ)

        self.assertEqual(selected, str(bundle))
        self.assertEqual(environ["SSL_CERT_FILE"], str(bundle))


if __name__ == "__main__":
    unittest.main()
