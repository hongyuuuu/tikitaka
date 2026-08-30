"""HTTP transport tests, driven by a fake opener. No network."""

from __future__ import annotations

import json
import socket
import unittest
import urllib.error

from tikitaka.models.base import (
    MalformedModelOutput,
    ModelRefused,
    ModelRoute,
    ModelTimeout,
    ModelUnavailable,
)
from tikitaka.models.http_transport import HttpTransport, HttpTransportConfig

ROUTE = ModelRoute(
    route_id="primary/gpt-5.6-terra",
    provider="openai",
    model="gpt-5.6-terra",
    reasoning_level="medium",
)
SECRET = "sk-not-a-real-key-0123456789"

# Shape copied from a real gpt-5.6-terra response.
LIVE_SHAPE = {
    "id": "chatcmpl-x",
    "model": "gpt-5.6-terra",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"ok":true}', "refusal": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 39,
        "completion_tokens": 148,
        "total_tokens": 187,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 77},
    },
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


class FakeOpener:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[object] = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return FakeResponse(self.result)


def transport(result: object, config: HttpTransportConfig | None = None) -> tuple:
    opener = FakeOpener(result)
    return HttpTransport(SECRET, ROUTE, config, opener=opener), opener


class RequestShapeTests(unittest.TestCase):
    def test_request_carries_model_prompt_and_reasoning_effort(self) -> None:
        client, opener = transport(LIVE_SHAPE)
        client.send("interpret this", {}, 30.0)

        request = opener.requests[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-5.6-terra")
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["messages"][0]["content"], "interpret this")
        self.assertEqual(request.get_method(), "POST")

    def test_credential_travels_only_in_the_auth_header(self) -> None:
        client, opener = transport(LIVE_SHAPE)
        client.send("hello", {}, 30.0)

        request = opener.requests[0]
        self.assertEqual(request.headers["Authorization"], f"Bearer {SECRET}")
        self.assertNotIn(SECRET, request.data.decode("utf-8"))
        self.assertNotIn(SECRET, repr(client))

    def test_response_format_is_off_unless_enabled(self) -> None:
        client, opener = transport(LIVE_SHAPE)
        client.send("hello", {"type": "object"}, 30.0)
        self.assertNotIn("response_format", json.loads(opener.requests[0].data))

        client, opener = transport(
            LIVE_SHAPE, HttpTransportConfig(use_response_format=True)
        )
        client.send("hello", {"type": "object"}, 30.0)
        body = json.loads(opener.requests[0].data)
        self.assertEqual(body["response_format"]["type"], "json_schema")


class ResponseParsingTests(unittest.TestCase):
    def test_live_shape_maps_onto_transport_response(self) -> None:
        client, _ = transport(LIVE_SHAPE)
        response = client.send("hi", {}, 30.0)

        self.assertEqual(response.text, '{"ok":true}')
        self.assertEqual(response.prompt_tokens, 39)
        self.assertEqual(response.completion_tokens, 148)
        self.assertEqual(response.reasoning_tokens, 77)
        # Reasoning is inside completion, never added to it.
        self.assertLess(response.reasoning_tokens, response.completion_tokens)

    def test_missing_usage_degrades_to_zeroes_not_an_error(self) -> None:
        payload = {"choices": [{"message": {"content": "{}"}}]}
        client, _ = transport(payload)
        response = client.send("hi", {}, 30.0)
        self.assertEqual(response.prompt_tokens, 0)
        self.assertEqual(response.reasoning_tokens, 0)

    def test_empty_choices_is_malformed(self) -> None:
        client, _ = transport({"choices": []})
        with self.assertRaises(MalformedModelOutput):
            client.send("hi", {}, 30.0)

    def test_refusal_is_surfaced_as_model_refused(self) -> None:
        payload = {
            "choices": [{"message": {"content": None, "refusal": "I cannot help"}}]
        }
        client, _ = transport(payload)
        with self.assertRaises(ModelRefused):
            client.send("hi", {}, 30.0)


class ErrorMappingTests(unittest.TestCase):
    def http_error(self, code: int) -> urllib.error.HTTPError:
        import io

        return urllib.error.HTTPError(
            "https://example.test", code, "err", {}, io.BytesIO(b'{"error":"x"}')
        )

    def test_rate_limit_and_server_faults_are_transient(self) -> None:
        for code in (429, 500, 503):
            with self.subTest(code=code):
                client, _ = transport(self.http_error(code))
                with self.assertRaises(ModelUnavailable):
                    client.send("hi", {}, 30.0)

    def test_auth_failure_is_refused_and_does_not_echo_the_key(self) -> None:
        client, _ = transport(self.http_error(401))
        with self.assertRaises(ModelRefused) as caught:
            client.send("hi", {}, 30.0)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_socket_timeout_maps_to_model_timeout(self) -> None:
        client, _ = transport(socket.timeout("too slow"))
        with self.assertRaises(ModelTimeout):
            client.send("hi", {}, 30.0)

    def test_url_error_maps_to_unavailable(self) -> None:
        client, _ = transport(urllib.error.URLError("no route to host"))
        with self.assertRaises(ModelUnavailable):
            client.send("hi", {}, 30.0)

    def test_wrapped_timeout_inside_url_error_is_still_a_timeout(self) -> None:
        client, _ = transport(urllib.error.URLError(socket.timeout("slow")))
        with self.assertRaises(ModelTimeout):
            client.send("hi", {}, 30.0)


if __name__ == "__main__":
    unittest.main()
