"""Minimal LLM clients over raw HTTP (httpx).

We deliberately avoid the anthropic/openai SDKs: appworld pins pydantic 1.x and the
current SDKs require pydantic 2.x. httpx is already an appworld dependency.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


_announced: set[str] = set()


def _announce(spec: str, endpoint, api_key: str) -> None:
    """One line per (spec, endpoint) pair, the first time that pair is constructed.

    Deduplicated because a sweep builds one client per cell and per pool run, and the point is to
    make the resolved endpoint auditable in a log, not to print it 30 times.
    """
    line = f"[llm] {spec} -> {endpoint}  key=…{api_key[-4:] if api_key else 'NONE'}"
    if line not in _announced:
        _announced.add(line)
        print(line, flush=True)


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def is_retryable(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def post_with_retry(client: httpx.Client, url: str, *, headers: dict, json: dict,
                    max_attempts: int = 10) -> httpx.Response:
    """POST with exponential backoff for transient HTTP failures."""
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.post(url, headers=headers, json=json)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if not is_retryable(status) or attempt == max_attempts:
                raise
            reason = f"HTTP {status}"
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                raise
            reason = type(exc).__name__
        print(f"    [llm retry {attempt}/{max_attempts - 1}] {reason}, "
              f"sleeping {delay:.0f}s", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


class BaseLLM:
    model: str = ""
    usage: LLMUsage

    def generate(self, system: str, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    """Anthropic Messages API — works with the official endpoint or any
    Anthropic-compatible relay via ANTHROPIC_BASE_URL (e.g. https://www.packyapi.com)."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (env or experiments/.env)")
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.usage = LLMUsage()
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=180.0)

        _announce(f"anthropic:{model}", self._client._merge_url("/v1/messages"), self.api_key)

    def generate(self, system, messages, max_tokens=2048, temperature=0.7):
        resp = post_with_retry(
            self._client, "/v1/messages",

            headers={"x-api-key": self.api_key,
                     "Authorization": f"Bearer {self.api_key}",
                     "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                  "system": system, "messages": messages},
        )
        data = resp.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("usage", {}).get("input_tokens", 0)
        self.usage.output_tokens += data.get("usage", {}).get("output_tokens", 0)
        return "".join(block["text"] for block in data["content"] if block["type"] == "text")


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-5-mini", api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set (env or experiments/.env)")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        self.usage = LLMUsage()
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=180.0)
        _announce(f"openai:{model}", self._client._merge_url("/v1/chat/completions"), self.api_key)

    def generate(self, system, messages, max_tokens=2048, temperature=0.7):
        resp = post_with_retry(
            self._client, "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                  "messages": [{"role": "system", "content": system}, *messages]},
        )
        data = resp.json()
        self.usage.calls += 1
        usage = data.get("usage", {})
        self.usage.input_tokens += usage.get("prompt_tokens", 0)
        self.usage.output_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


class MockLLM(BaseLLM):
    """Returns queued responses; for unit tests of the agent loop."""

    def __init__(self, responses: list[str] | None = None):
        self.model = "mock"
        self.usage = LLMUsage()
        self.responses = list(responses or [])
        self.seen: list[dict] = field(default_factory=list) if False else []

    def generate(self, system, messages, max_tokens=2048, temperature=0.7):
        self.usage.calls += 1
        self.seen.append({"system": system, "messages": messages})
        if not self.responses:
            return "```python\napis.supervisor.complete_task()\n```"
        return self.responses.pop(0)


def build_llm(spec: str) -> BaseLLM:
    """spec: 'anthropic:claude-sonnet-5' | 'openai:gpt-5-mini' | 'mock'."""
    if spec == "mock":
        return MockLLM()
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicLLM(model=model)
    if provider == "openai":
        return OpenAILLM(model=model)
    raise ValueError(f"unknown llm spec: {spec}")
