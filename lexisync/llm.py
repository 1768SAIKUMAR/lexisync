"""Stage 4a - LLM backends: Ollama (local, default), any OpenAI-compatible API,
and a deterministic mock for tests and offline demos.

Determinism: temperature 0 (always take the most likely token) + fixed seed,
so the same question + passages give the same answer - needed for a fair
evaluation and a reproducible demo.

Structured output: we pass the JSON schema to Ollama's `format` field, which
constrains decoding to valid JSON of that shape. The *order* of the fields in
the schema is the order the model writes them: evidence (claims + quotes)
first, the decision to abstain last - so the model commits to evidence before
it decides, instead of deciding first and inventing support afterwards.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import requests

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["text", "citations", "quote"],
            },
        },
        "answer": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
        "explanation": {"type": "string"},
        "conflicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims", "answer", "missing", "abstain", "explanation", "conflicts"],
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0


class LLM(Protocol):
    name: str
    def generate(self, system: str, user: str, schema: dict | None = None) -> LLMResponse: ...


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434",
                 seed: int = 42, num_ctx: int = 8192, timeout: int = 180,
                 max_tokens: int = 1024):
        self.name = f"ollama:{model}"
        self.model, self.host, self.seed = model, host.rstrip("/"), seed
        self.num_ctx, self.timeout, self.max_tokens = num_ctx, timeout, max_tokens

    def generate(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                # num_predict caps the answer: JSON-constrained decoding can get stuck
                # emitting whitespace forever; a normal answer needs ~350 tokens.
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx,
                            "num_predict": self.max_tokens}}
        if schema:
            body["format"] = schema
        t = time.perf_counter()
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise RuntimeError("Ollama is not running. Start it with "
                               "`brew services start ollama` (or `ollama serve`).") from e
        except requests.Timeout as e:
            raise RuntimeError(f"Ollama did not answer within {self.timeout} s.") from e
        r.raise_for_status()
        j = r.json()
        return LLMResponse(j["message"]["content"], j.get("prompt_eval_count", 0),
                           j.get("eval_count", 0), (time.perf_counter() - t) * 1000)


class OpenAICompatLLM:
    """Any /v1/chat/completions API (OpenAI, Groq, Together, vLLM, LM Studio...)."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 seed: int = 42, timeout: int = 120):
        self.name = f"openai:{model}"
        self.model, self.base_url, self.seed, self.timeout = model, base_url.rstrip("/"), seed, timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        body = {"model": self.model, "temperature": 0, "seed": self.seed, "max_tokens": 1024,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if schema:
            body["response_format"] = {"type": "json_object"}
        t = time.perf_counter()
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return LLMResponse(j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0),
                           u.get("completion_tokens", 0), (time.perf_counter() - t) * 1000)


class MockLLM:
    """Deterministic stand-in. With `script`, returns those responses in order
    (tests of the verifier/retry loop). Without, it answers extractively: one
    claim per relevant passage, quoting that passage's first words - correct
    by construction, so the rest of the pipeline can be demoed offline."""
    name = "mock"

    def __init__(self, script: list | None = None):
        self.script = list(script or [])
        self.calls: list[str] = []

    def generate(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        self.calls.append(user)
        if self.script:
            out = self.script.pop(0)
            if callable(out):                  # a function of the prompt
                out = out(user)
            return LLMResponse(out if isinstance(out, str) else json.dumps(out), 100, 50, 1.0)
        passages = re.findall(r"^\[(C\d+)\] [^\n]*\n(.*?)(?=^\[C\d+\] |\Z)", user, re.S | re.M)
        if not passages:
            return LLMResponse(json.dumps({"claims": [], "answer": "", "missing": [],
                                           "abstain": True, "explanation": "no passages",
                                           "conflicts": []}), 100, 30, 1.0)
        claims = []
        for cid, text in passages[:2]:
            words = text.split()
            quote = " ".join(words[:12])
            claims.append({"text": " ".join(words[:25]), "citations": [cid], "quote": quote})
        ans = " ".join(f"{c['text']} [{c['citations'][0]}]" for c in claims)
        return LLMResponse(json.dumps({"claims": claims, "answer": ans, "missing": [],
                                       "abstain": False, "explanation": "extractive mock",
                                       "conflicts": []}), 100, 80, 1.0)


def get_llm(cfg: dict) -> LLM:
    c = cfg["llm"]
    backend = os.environ.get("LEXISYNC_LLM", c["backend"])
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAICompatLLM(c["openai_model"], c["openai_base_url"], seed=cfg["seed"])
    return OllamaLLM(c["ollama_model"], c["ollama_host"], seed=cfg["seed"], num_ctx=c["num_ctx"],
                     timeout=c.get("timeout_s", 180), max_tokens=c.get("max_tokens", 1024))
