import os
import httpx
from openai import OpenAI
from chutes_e2ee import ChutesE2EETransport

API_KEY = os.environ["CHUTES_API_KEY"]
MODEL = "deepseek-ai/DeepSeek-V3.1-TEE"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://llm.chutes.ai/v1",
    http_client=httpx.Client(
        transport=ChutesE2EETransport(api_key=API_KEY),
    ),
)

print("=== Non-streaming ===")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
)
print(resp.choices[0].message.content)

print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
