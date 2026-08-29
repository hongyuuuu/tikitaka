from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.retrieval_fakes import SemanticFakeEmbedder
from tikitaka.retrieval.catalog import load_catalog
from tikitaka.retrieval.dense import (
    DenseArtifactError,
    DenseRouteError,
    build_dense_artifact,
    embed_query_for_index,
    load_dense_index,
    load_dense_index_safe,
    normalize_embedding,
)
from tikitaka.retrieval.manifests import DENSE_CHECKPOINT_FILENAME
from tikitaka.retrieval.text import build_dense_query


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_small.jsonl"


class DenseRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(FIXTURE)

    def _build(self, directory: str, embedder: object | None = None):
        selected = embedder or SemanticFakeEmbedder()
        manifest = build_dense_artifact(
            self.catalog,
            selected,
            directory,
            embedding_provider="fixture",
            embedding_model="semantic-keywords-v1",
            batch_size=2,
        )
        return selected, manifest

    def test_build_load_and_semantic_paraphrase_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embedder, manifest = self._build(directory)
            index = load_dense_index(
                directory,
                self.catalog,
                embedding_route_id=embedder.route_id,
                index_id=manifest.index_id,
            )
            query = build_dense_query("rainproof mountain footwear for a trip")
            vector = embed_query_for_index(embedder, index, query)
            hits = index.search(vector, limit=3)

        self.assertEqual(manifest.document_count, 7)
        self.assertEqual(manifest.embedding_dimension, 8)
        self.assertIn(index.backend, {"python-exact", "numpy-exact"})
        self.assertIn("A_HIKE", [hit.parent_asin for hit in hits[:2]])
        self.assertEqual(len({hit.parent_asin for hit in hits}), len(hits))
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_build_resumes_only_with_matching_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            flaky = SemanticFakeEmbedder(fail_after_document_calls=1)
            with self.assertRaisesRegex(DenseRouteError, "catalog offset 2"):
                self._build(directory, flaky)
            checkpoint = Path(directory) / DENSE_CHECKPOINT_FILENAME
            progress = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(progress["next_index"], 2)

            stable, manifest = self._build(directory, SemanticFakeEmbedder())
            index = load_dense_index(directory, self.catalog)
            self.assertFalse(checkpoint.exists())

        self.assertEqual(index.manifest.index_id, manifest.index_id)
        self.assertEqual(stable.route_id, manifest.embedding_route_id)

    def test_route_index_and_checksum_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._build(directory)
            with self.assertRaisesRegex(DenseArtifactError, "embedding_route_id"):
                load_dense_index(directory, self.catalog, embedding_route_id="wrong-route")
            with self.assertRaisesRegex(DenseArtifactError, "index_id"):
                load_dense_index(directory, self.catalog, index_id="wrong-index")
            vectors = Path(directory) / "vectors.f32"
            with vectors.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(DenseArtifactError, "checksum mismatch"):
                load_dense_index(directory, self.catalog, index_id=manifest.index_id)

    def test_artifact_identity_and_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            _, first = self._build(first_directory)
            _, second = self._build(second_directory)
        self.assertEqual(first.index_id, second.index_id)
        self.assertEqual(dict(first.artifact_checksums), dict(second.artifact_checksums))

    def test_catalog_source_mismatch_is_rejected_even_with_same_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as catalog_directory:
            self._build(directory)
            altered_path = Path(catalog_directory) / "catalog.jsonl"
            altered_path.write_text(
                "".join(line.rstrip("\n") + "  \n" for line in FIXTURE.read_text(encoding="utf-8").splitlines(True)),
                encoding="utf-8",
            )
            altered_catalog = load_catalog(altered_path, expected_count=7)
            with self.assertRaisesRegex(DenseArtifactError, "catalog_source_sha256"):
                load_dense_index(directory, altered_catalog)

    def test_vector_validation_rejects_zero_nonfinite_and_wrong_dimension(self) -> None:
        with self.assertRaisesRegex(DenseRouteError, "non-zero norm"):
            normalize_embedding((0.0, 0.0))
        with self.assertRaisesRegex(DenseRouteError, "finite"):
            normalize_embedding((1.0, float("nan")))
        with self.assertRaisesRegex(DenseRouteError, "dimension mismatch"):
            normalize_embedding((1.0, 2.0), expected_dimension=3)
        with self.assertRaisesRegex(DenseRouteError, "non-zero norm"):
            normalize_embedding((1e308, 1e308))

    def test_manifest_types_and_derived_identity_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._build(directory)
            manifest_path = Path(directory) / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["embedding_dimension"] = True
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DenseArtifactError, "invalid field types"):
                load_dense_index(directory, self.catalog)

        with tempfile.TemporaryDirectory() as directory:
            self._build(directory)
            manifest_path = Path(directory) / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["embedding_model"] = "tampered-model"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DenseArtifactError, "derived_index_id"):
                load_dense_index(directory, self.catalog)

    def test_builder_recovers_interrupted_final_file_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, original = self._build(directory)
            root = Path(directory)
            (root / "manifest.json").unlink()
            (root / "vectors.f32").replace(root / "vectors.f32.partial")
            checkpoint = {
                "checkpoint_version": 1,
                "artifact_format_version": "dense-f32-v1",
                "catalog_source_sha256": self.catalog.identity.source_sha256,
                "catalog_row_count": len(self.catalog),
                "ordered_parent_asin_sha256": self.catalog.identity.ordered_parent_asin_sha256,
                "product_text_schema_version": "product_text_v1",
                "embedding_provider": "fixture",
                "embedding_model": "semantic-keywords-v1",
                "embedding_route_id": "fixture-semantic-v1",
                "batch_size": 2,
                "next_index": len(self.catalog),
                "embedding_dimension": 8,
            }
            (root / DENSE_CHECKPOINT_FILENAME).write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )
            _, recovered = self._build(directory)
            index = load_dense_index(directory, self.catalog)

        self.assertEqual(recovered.index_id, original.index_id)
        self.assertEqual(index.manifest.index_id, original.index_id)

    def test_safe_loader_makes_missing_or_corrupt_dense_artifacts_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = load_dense_index_safe(directory, self.catalog)
            self.assertIsNone(missing.index)
            self.assertEqual(missing.failure_code, "dense_artifact_unavailable")

        with tempfile.TemporaryDirectory() as directory:
            self._build(directory)
            with (Path(directory) / "vectors.f32").open("ab") as handle:
                handle.write(b"corrupt")
            corrupt = load_dense_index_safe(directory, self.catalog)
            self.assertIsNone(corrupt.index)
            self.assertEqual(corrupt.failure_code, "dense_artifact_invalid")


if __name__ == "__main__":
    unittest.main()
