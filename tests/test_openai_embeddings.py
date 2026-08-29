from __future__ import annotations

import io
import json
import socket
import unittest
import urllib.error

from tikitaka.models.base import (
    CredentialMissing,
    MalformedModelOutput,
    ModelRefused,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
)
from tikitaka.retrieval.openai_embeddings import (
    MAX_INPUTS_PER_REQUEST,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
    openai_embedder_from_env,
    openai_embedding_route,
)


SECRET = "sk-test-secret-that-must-never-leak"


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


class RecordingOpener:
    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        selected = self.script.pop(0)
        if isinstance(selected, Exception):
            raise selected
        return FakeResponse(selected)


def response(*vectors: list[float], prompt_tokens: int = 12) -> dict:
    return {
        "object": "list",
        "model": "text-embedding-3-large",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.openai.com/v1/embeddings",
        status,
        "error",
        {},
        io.BytesIO(b'{"error":{"message":"provider detail"}}'),
    )


class OpenAIEmbeddingTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OpenAIEmbeddingConfig(
            timeout_s=7.0,
            max_attempts=2,
            backoff_base_s=0.01,
        )
        self.route = openai_embedding_route(self.config)

    def test_batch_request_preserves_index_order_and_attributes_usage(self) -> None:
        payload = response([1.0, 0.0], [0.0, 1.0], prompt_tokens=100)
        payload["data"].reverse()
        opener = RecordingOpener(payload)
        model = OpenAIEmbeddingModel(SECRET, self.config, opener=opener)

        vectors, usage = model.embed(("first", "second"), self.route)

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.route, self.route.route_id)
        self.assertAlmostEqual(usage.estimated_cost, 0.000013)
        body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(body["input"], ["first", "second"])
        self.assertEqual(body["model"], "text-embedding-3-large")
        self.assertEqual(body["encoding_format"], "float")
        self.assertNotIn("dimensions", body)
        self.assertEqual(opener.timeouts, [7.0])
        self.assertNotIn(SECRET, repr(model))

    def test_dimensions_are_part_of_request_and_route_identity(self) -> None:
        config = OpenAIEmbeddingConfig(dimensions=1)
        route = openai_embedding_route(config)
        opener = RecordingOpener(response([1.0]))
        model = OpenAIEmbeddingModel(SECRET, config, opener=opener)

        model.embed(("one",), route)

        body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(body["dimensions"], 1)
        self.assertEqual(route.route_id, "openai/text-embedding-3-large/dimensions-1")

    def test_transient_status_retries_with_bounded_backoff(self) -> None:
        pauses: list[float] = []
        opener = RecordingOpener(http_error(429), response([1.0]))
        model = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=opener,
            sleep=pauses.append,
        )

        vectors, usage = model.embed(("retry",), self.route)

        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(usage.calls, 1)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(pauses, [0.01])

    def test_exhausted_transient_status_is_attributable(self) -> None:
        opener = RecordingOpener(http_error(503), http_error(503))
        model = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=opener,
            sleep=lambda _: None,
        )
        with self.assertRaises(ModelUnavailable) as caught:
            model.embed(("retry",), self.route)
        self.assertEqual(caught.exception.route, self.route)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_timeout_and_authentication_fail_closed_without_secret(self) -> None:
        timeout = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(socket.timeout()),
        )
        with self.assertRaises(ModelTimeout):
            timeout.embed(("timeout",), self.route)

        refused = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(http_error(401)),
        )
        with self.assertRaises(ModelRefused) as caught:
            refused.embed(("auth",), self.route)
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertNotIn("provider detail", str(caught.exception))

    def test_malformed_response_and_identity_mismatch_fail_closed(self) -> None:
        malformed = response([1.0], [2.0])
        malformed["data"][1]["index"] = 0
        model = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(malformed),
        )
        with self.assertRaisesRegex(MalformedModelOutput, "indexes"):
            model.embed(("one", "two"), self.route)

        wrong_route = ModelRoute(
            route_id="openai/text-embedding-3-small/dimensions-default",
            provider="openai",
            model="text-embedding-3-small",
        )
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            model.embed(("one",), wrong_route)

        wrong_count = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(response([1.0])),
        )
        with self.assertRaisesRegex(MalformedModelOutput, "number of vectors"):
            wrong_count.embed(("one", "two"), self.route)

        missing_usage = response([1.0])
        missing_usage.pop("usage")
        no_usage = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(missing_usage),
        )
        with self.assertRaisesRegex(MalformedModelOutput, "no usage"):
            no_usage.embed(("one",), self.route)

    def test_empty_oversized_and_invalid_inputs_are_rejected_locally(self) -> None:
        model = OpenAIEmbeddingModel(
            SECRET,
            self.config,
            opener=RecordingOpener(response([1.0])),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.embed((), self.route)
        with self.assertRaisesRegex(ValueError, "sequence"):
            model.embed("not-a-sequence", self.route)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            model.embed((" ",), self.route)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            model.embed(tuple("x" for _ in range(MAX_INPUTS_PER_REQUEST + 1)), self.route)


class OpenAIEmbeddingFactoryTests(unittest.TestCase):
    def test_factory_builds_gateway_embedder_from_explicit_environment(self) -> None:
        opener = RecordingOpener(response([1.0, 0.0]))
        embedder = openai_embedder_from_env(
            {
                "OPENAI_API_KEY": SECRET,
                "TIKITAKA_EMBEDDING_DIMENSIONS": "2",
                "TIKITAKA_EMBEDDING_MAX_ATTEMPTS": "1",
                "TIKITAKA_EMBEDDING_INPUT_COST_PER_1M": "0.13",
            },
            opener=opener,
        )

        vector = embedder.embed_query("walking shoes")

        self.assertEqual(vector, (1.0, 0.0))
        self.assertEqual(embedder.provider, "openai")
        self.assertEqual(embedder.model, "text-embedding-3-large")
        self.assertIn("dimensions-2", embedder.route_id)
        self.assertEqual(embedder.usage.calls, 1)
        self.assertNotIn(SECRET, repr(embedder))

    def test_missing_credential_and_invalid_environment_are_rejected(self) -> None:
        with self.assertRaises(CredentialMissing):
            openai_embedder_from_env({})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            openai_embedder_from_env(
                {
                    "OPENAI_API_KEY": SECRET,
                    "TIKITAKA_EMBEDDING_DIMENSIONS": "large",
                }
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            OpenAIEmbeddingConfig(dimensions=0)


if __name__ == "__main__":
    unittest.main()
