"""
httpx Transports that transparently add Chutes E2EE encryption.

Usage with the OpenAI SDK:

    from openai import OpenAI
    from chutes_e2ee import ChutesE2EETransport

    client = OpenAI(
        api_key="cpk_...",
        base_url="https://llm.chutes.ai/v1",  # OpenAI SDK uses this for /v1/models
        http_client=httpx.Client(transport=ChutesE2EETransport(api_key="cpk_...")),
    )
    # The transport routes /v1/models to llm.chutes.ai and all E2EE
    # calls (/e2e/invoke, /e2e/instances) to api.chutes.ai.
    resp = client.chat.completions.create(model="deepseek-ai/DeepSeek-R1", messages=[...])
"""

from __future__ import annotations

import json
import typing
from urllib.request import getproxies

import httpx

from chutes_e2ee.crypto import (
    build_e2ee_request,
    decrypt_response,
    decrypt_stream_chunk,
    decrypt_stream_init,
)
from chutes_e2ee.discovery import DiscoveryManager

_DEFAULT_API_BASE = "https://api.chutes.ai"
_DEFAULT_MODELS_BASE = "https://llm.chutes.ai"


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


def _read_proxy_env() -> tuple[str | None, str | None, list[str]]:
    # getproxies() reads env vars (HTTP_PROXY, HTTPS_PROXY, NO_PROXY) cross-platform,
    # including Windows Registry and macOS System Configuration.
    proxies = getproxies()
    no_proxy = [h.strip() for h in proxies.get("no", "").split(",") if h.strip()]
    return proxies.get("https"), proxies.get("http"), no_proxy


def _matches_no_proxy(host: str, no_proxy: list[str]) -> bool:
    # Entries are pre-normalised (lowercased, leading dot stripped) at init time.
    host = host.lower()
    for pattern in no_proxy:
        if pattern == "*" or host == pattern or host.endswith("." + pattern):
            return True
    return False


class _EnvProxyTransport(httpx.BaseTransport):
    def __init__(
        self,
        https_proxy: str | None,
        http_proxy: str | None,
        no_proxy: list[str],
    ) -> None:
        self._no_proxy = [e.lower().lstrip(".") for e in no_proxy]
        self._direct = httpx.HTTPTransport()
        self._https = httpx.HTTPTransport(proxy=https_proxy) if https_proxy else None
        self._http = httpx.HTTPTransport(proxy=http_proxy) if http_proxy else None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if _matches_no_proxy(request.url.host, self._no_proxy):
            return self._direct.handle_request(request)
        if request.url.scheme == "https" and self._https:
            return self._https.handle_request(request)
        if self._http:
            return self._http.handle_request(request)
        return self._direct.handle_request(request)

    def close(self) -> None:
        self._direct.close()
        if self._https:
            self._https.close()
        if self._http:
            self._http.close()


class _AsyncEnvProxyTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        https_proxy: str | None,
        http_proxy: str | None,
        no_proxy: list[str],
    ) -> None:
        self._no_proxy = [e.lower().lstrip(".") for e in no_proxy]
        self._direct = httpx.AsyncHTTPTransport()
        self._https = httpx.AsyncHTTPTransport(proxy=https_proxy) if https_proxy else None
        self._http = httpx.AsyncHTTPTransport(proxy=http_proxy) if http_proxy else None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if _matches_no_proxy(request.url.host, self._no_proxy):
            return await self._direct.handle_async_request(request)
        if request.url.scheme == "https" and self._https:
            return await self._https.handle_async_request(request)
        if self._http:
            return await self._http.handle_async_request(request)
        return await self._direct.handle_async_request(request)

    async def aclose(self) -> None:
        await self._direct.aclose()
        if self._https:
            await self._https.aclose()
        if self._http:
            await self._http.aclose()


def _default_transport() -> httpx.BaseTransport:
    https_proxy, http_proxy, no_proxy = _read_proxy_env()
    if not (https_proxy or http_proxy):
        return httpx.HTTPTransport()
    if no_proxy or (https_proxy and http_proxy and https_proxy != http_proxy):
        return _EnvProxyTransport(https_proxy, http_proxy, no_proxy)
    return httpx.HTTPTransport(proxy=https_proxy or http_proxy)


def _default_async_transport() -> httpx.AsyncBaseTransport:
    https_proxy, http_proxy, no_proxy = _read_proxy_env()
    if not (https_proxy or http_proxy):
        return httpx.AsyncHTTPTransport()
    if no_proxy or (https_proxy and http_proxy and https_proxy != http_proxy):
        return _AsyncEnvProxyTransport(https_proxy, http_proxy, no_proxy)
    return httpx.AsyncHTTPTransport(proxy=https_proxy or http_proxy)


def _extract_json_body(request: httpx.Request) -> dict | None:
    """Try to parse the request body as JSON."""
    body = request.content
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _extract_model(payload: dict) -> str | None:
    return payload.get("model")


def _is_streaming(payload: dict) -> bool:
    return bool(payload.get("stream", False))


def _original_path(request: httpx.Request) -> str:
    """Return the path component of the request URL (e.g. /v1/chat/completions)."""
    return request.url.raw_path.decode("ascii").split("?")[0]


def _build_invoke_headers(
    api_key: str,
    chute_id: str,
    instance_id: str,
    nonce: str,
    stream: bool,
    e2e_path: str,
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Chute-Id": chute_id,
        "X-Instance-Id": instance_id,
        "X-E2E-Nonce": nonce,
        "X-E2E-Stream": str(stream).lower(),
        "X-E2E-Path": e2e_path,
        "Content-Type": "application/octet-stream",
    }


def _build_json_response(data: dict, request: httpx.Request) -> httpx.Response:
    """Construct a synthetic httpx.Response with JSON content."""
    body = json.dumps(data).encode()
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=body,
        request=request,
    )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _iter_sse_from_e2ee(
    raw_stream: typing.Iterator[bytes], response_sk: bytes
) -> typing.Iterator[bytes]:
    """Consume encrypted SSE events and yield decrypted standard SSE lines."""
    stream_key: bytes | None = None
    buffer = b""

    for chunk in raw_stream:
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            yield from _process_sse_line(line, response_sk, stream_key_holder := [stream_key])
            stream_key = stream_key_holder[0]


def _process_sse_line(
    line: str,
    response_sk: bytes,
    stream_key_holder: list[bytes | None],
) -> typing.Iterator[bytes]:
    """Process a single SSE line and yield any output bytes."""
    if not line.startswith("data: "):
        # Drop empty lines and non-data lines from the encrypted stream;
        # decrypted output lines already include their own SSE framing.
        return

    raw = line[6:].strip()
    if raw == "[DONE]":
        yield b"data: [DONE]\n\n"
        return

    if not raw:
        return

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return

    stream_key = stream_key_holder[0]

    if "e2e_init" in event:
        stream_key_holder[0] = decrypt_stream_init(response_sk, event["e2e_init"])

    elif "e2e" in event:
        if stream_key is None:
            raise RuntimeError("Received e2e chunk before e2e_init")
        decrypted = decrypt_stream_chunk(event["e2e"], stream_key)
        yield f"{decrypted}\n\n".encode()

    elif "usage" in event:
        # Usage events are plaintext — pass through as-is.
        yield (line + "\n\n").encode()

    elif "e2e_error" in event:
        error_data = json.dumps({"error": event["e2e_error"]})
        yield f"data: {error_data}\n\n".encode()


async def _aiter_sse_from_e2ee(
    raw_stream: typing.AsyncIterator[bytes], response_sk: bytes
) -> typing.AsyncIterator[bytes]:
    """Async variant: consume encrypted SSE events and yield decrypted standard SSE lines."""
    stream_key: bytes | None = None
    buffer = b""

    async for chunk in raw_stream:
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            for output in _process_sse_line(line, response_sk, stream_key_holder := [stream_key]):
                yield output
            stream_key = stream_key_holder[0]


# ---------------------------------------------------------------------------
# Sync Transport
# ---------------------------------------------------------------------------


class ChutesE2EETransport(httpx.BaseTransport):
    """Synchronous httpx transport that wraps requests in Chutes E2EE encryption.

    Intercepts outgoing HTTP requests, encrypts the JSON payload using ML-KEM-768
    + HKDF + ChaCha20-Poly1305, routes them through ``/e2e/invoke``, and
    transparently decrypts the response so the caller sees a normal JSON response.

    Args:
        api_key: Chutes API key (``cpk_...``).
        api_base: Base URL for E2EE API calls (default: ``https://api.chutes.ai``).
        models_base: Base URL for the ``/v1/models`` listing endpoint
                     (default: ``https://llm.chutes.ai``).
        inner: Optional underlying httpx transport for the actual HTTP calls.
               Defaults to ``httpx.HTTPTransport()``.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = _DEFAULT_API_BASE,
        models_base: str = _DEFAULT_MODELS_BASE,
        inner: httpx.BaseTransport | None = None,
    ):
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._inner = inner if inner is not None else _default_transport()
        self._discovery = DiscoveryManager(
            api_base=self._api_base, models_base=models_base, api_key=api_key
        )
        # Lazy client for discovery calls (reuses the inner transport).
        self._http: httpx.Client | None = None

    def _get_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(transport=self._inner)
        return self._http

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        payload = _extract_json_body(request)
        if payload is None:
            # Not a JSON request — pass through (e.g. /v1/models).
            return self._inner.handle_request(request)

        model = _extract_model(payload)
        if model is None:
            return self._inner.handle_request(request)

        stream = _is_streaming(payload)
        e2e_path = _original_path(request)
        http = self._get_http()

        # Resolve model -> chute_id.
        chute_id = self._discovery.resolve_chute_id(model, http)

        # Get a nonce (fetches fresh instances if needed).
        instance, nonce = self._discovery.get_nonce(chute_id, http)

        # Build encrypted blob.
        result = build_e2ee_request(instance.e2e_pubkey, payload)

        # Build the actual request to /e2e/invoke.
        headers = _build_invoke_headers(
            self._api_key, chute_id, instance.instance_id, nonce, stream, e2e_path
        )
        invoke_url = f"{self._api_base}/e2e/invoke"

        if stream:
            return self._handle_stream(
                invoke_url, headers, result.blob, result.response_sk, request
            )
        else:
            return self._handle_non_stream(
                invoke_url, headers, result.blob, result.response_sk, request
            )

    def _handle_non_stream(
        self,
        url: str,
        headers: dict,
        blob: bytes,
        response_sk: bytes,
        original_request: httpx.Request,
    ) -> httpx.Response:
        invoke_request = httpx.Request("POST", url, headers=headers, content=blob)
        response = self._inner.handle_request(invoke_request)

        if response.status_code != 200:
            # Surface the error transparently.
            response._request = original_request
            return response

        # Read the full response body for non-streaming.
        body = b""
        for chunk in response.stream:
            body += chunk
        response.close()

        decrypted = decrypt_response(body, response_sk)
        return _build_json_response(decrypted, original_request)

    def _handle_stream(
        self,
        url: str,
        headers: dict,
        blob: bytes,
        response_sk: bytes,
        original_request: httpx.Request,
    ) -> httpx.Response:
        invoke_request = httpx.Request("POST", url, headers=headers, content=blob)
        response = self._inner.handle_request(invoke_request)

        if response.status_code != 200:
            response._request = original_request
            return response

        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=_SyncDecryptedStream(response.stream, response_sk),
            request=original_request,
        )

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
        self._inner.close()


class _SyncDecryptedStream(httpx.SyncByteStream):
    """Wraps an encrypted SSE sync byte stream, yielding decrypted SSE bytes."""

    def __init__(self, raw_stream: httpx.SyncByteStream, response_sk: bytes):
        self._raw_stream = raw_stream
        self._response_sk = response_sk

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from _iter_sse_from_e2ee(iter(self._raw_stream), self._response_sk)

    def close(self) -> None:
        self._raw_stream.close()


# ---------------------------------------------------------------------------
# Async Transport
# ---------------------------------------------------------------------------


class AsyncChutesE2EETransport(httpx.AsyncBaseTransport):
    """Asynchronous httpx transport that wraps requests in Chutes E2EE encryption.

    Same as :class:`ChutesE2EETransport` but for ``httpx.AsyncClient`` /
    ``AsyncOpenAI``.

    Args:
        api_key: Chutes API key (``cpk_...``).
        api_base: Base URL for E2EE API calls (default: ``https://api.chutes.ai``).
        models_base: Base URL for the ``/v1/models`` listing endpoint
                     (default: ``https://llm.chutes.ai``).
        inner: Optional underlying async httpx transport. Defaults to
               ``httpx.AsyncHTTPTransport()``.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = _DEFAULT_API_BASE,
        models_base: str = _DEFAULT_MODELS_BASE,
        inner: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._inner = inner if inner is not None else _default_async_transport()
        self._discovery = DiscoveryManager(
            api_base=self._api_base, models_base=models_base, api_key=api_key
        )
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(transport=self._inner)
        return self._http

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = _extract_json_body(request)
        if payload is None:
            return await self._inner.handle_async_request(request)

        model = _extract_model(payload)
        if model is None:
            return await self._inner.handle_async_request(request)

        stream = _is_streaming(payload)
        e2e_path = _original_path(request)
        http = await self._get_http()

        chute_id = await self._discovery.resolve_chute_id_async(model, http)
        instance, nonce = await self._discovery.get_nonce_async(chute_id, http)

        result = build_e2ee_request(instance.e2e_pubkey, payload)

        headers = _build_invoke_headers(
            self._api_key, chute_id, instance.instance_id, nonce, stream, e2e_path
        )
        invoke_url = f"{self._api_base}/e2e/invoke"

        if stream:
            return await self._handle_stream(
                invoke_url, headers, result.blob, result.response_sk, request
            )
        else:
            return await self._handle_non_stream(
                invoke_url, headers, result.blob, result.response_sk, request
            )

    async def _handle_non_stream(
        self,
        url: str,
        headers: dict,
        blob: bytes,
        response_sk: bytes,
        original_request: httpx.Request,
    ) -> httpx.Response:
        invoke_request = httpx.Request("POST", url, headers=headers, content=blob)
        response = await self._inner.handle_async_request(invoke_request)

        if response.status_code != 200:
            response._request = original_request
            return response

        # Read the full response body for non-streaming.
        body = b""
        async for chunk in response.stream:
            body += chunk
        await response.aclose()

        decrypted = decrypt_response(body, response_sk)
        return _build_json_response(decrypted, original_request)

    async def _handle_stream(
        self,
        url: str,
        headers: dict,
        blob: bytes,
        response_sk: bytes,
        original_request: httpx.Request,
    ) -> httpx.Response:
        invoke_request = httpx.Request("POST", url, headers=headers, content=blob)
        response = await self._inner.handle_async_request(invoke_request)

        if response.status_code != 200:
            response._request = original_request
            return response

        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncDecryptedStream(response.stream, response_sk),
            request=original_request,
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        await self._inner.aclose()


class _AsyncDecryptedStream(httpx.AsyncByteStream):
    """Wraps an encrypted SSE async byte stream, yielding decrypted SSE bytes."""

    def __init__(self, raw_stream: httpx.AsyncByteStream, response_sk: bytes):
        self._raw_stream = raw_stream
        self._response_sk = response_sk

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in _aiter_sse_from_e2ee(self._raw_stream.__aiter__(), self._response_sk):
            yield chunk

    async def aclose(self) -> None:
        await self._raw_stream.aclose()
