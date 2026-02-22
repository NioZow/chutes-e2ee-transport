# chutes-e2ee

End-to-end encrypted transport for [Chutes AI](https://chutes.ai), designed as a drop-in `httpx` transport for the [OpenAI Python SDK](https://github.com/openai/openai-python).

Requests are encrypted client-side using **ML-KEM-768 + HKDF-SHA256 + ChaCha20-Poly1305** so that neither the API relay nor any intermediary can read the payload — only the GPU instance running your model sees the plaintext.

## Installation

```bash
pip install chutes-e2ee
```

## Quick Start

### Synchronous (OpenAI)

```python
import httpx
from openai import OpenAI
from chutes_e2ee import ChutesE2EETransport

API_KEY = "cpk_..."

client = OpenAI(
    api_key=API_KEY,
    base_url="https://llm.chutes.ai/v1",
    http_client=httpx.Client(
        transport=ChutesE2EETransport(api_key=API_KEY, api_base="https://llm.chutes.ai"),
    ),
)

response = client.chat.completions.create(
    model="zai-org/GLM-4.7-TEE",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Async (AsyncOpenAI)

```python
import httpx
from openai import AsyncOpenAI
from chutes_e2ee import AsyncChutesE2EETransport

API_KEY = "cpk_..."

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://llm.chutes.ai/v1",
    http_client=httpx.AsyncClient(
        transport=AsyncChutesE2EETransport(api_key=API_KEY, api_base="https://llm.chutes.ai"),
    ),
)

response = await client.chat.completions.create(
    model="zai-org/GLM-4.7-TEE",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Streaming

Streaming works transparently — the transport decrypts chunks on-the-fly:

```python
stream = client.chat.completions.create(
    model="zai-org/GLM-4.7-TEE",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## How It Works

The transport intercepts requests at the `httpx` layer — completely invisible to the OpenAI SDK:

1. **Outbound**: Parses the JSON body, discovers E2EE-capable instances (caching pubkeys + nonces), performs ML-KEM encapsulation + HKDF key derivation + ChaCha20-Poly1305 encryption, and rewrites the request to `POST /e2e/invoke` with a binary body and E2EE headers.

2. **Inbound (non-streaming)**: Decrypts the response blob (ML-KEM decapsulation + HKDF + ChaCha20) and returns a normal JSON `httpx.Response`.

3. **Inbound (streaming)**: Decrypts the `e2e_init` key-exchange event, then decrypts each `e2e` chunk, yielding standard `data: {...}` SSE lines that the OpenAI SDK parses normally.

### Nonce Management

The transport automatically:
- Prefetches nonces from `/e2e/instances/{chute_id}` (10 per instance, up to 5 instances)
- Caches them locally with expiry tracking (60s client TTL)
- Refreshes when nonces are exhausted or expired
- Consumes one nonce per request (single-use, replay-proof)

### Model Resolution

You can use model names (e.g. `zai-org/GLM-4.7-TEE`) or chute IDs directly. The transport resolves model names to chute IDs via the `/v1/models` endpoint (cached for 5 minutes).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | *required* | Your Chutes API key |
| `api_base` | `https://api.chutes.ai` | API base URL |
| `inner` | `httpx.HTTPTransport()` | Underlying transport for actual HTTP calls |

## Dependencies

- `httpx` >= 0.25
- `cryptography` >= 42.0
- `pqcrypto` >= 0.1.0 (ML-KEM-768 implementation)

## License

MIT
