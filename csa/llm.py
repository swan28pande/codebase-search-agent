"""Thin async client for a vLLM server (OpenAI-compatible API)."""
from __future__ import annotations

import os

import httpx


class VLLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 60):
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.http = httpx.AsyncClient(timeout=timeout)

    async def chat(self, system: str, user: str, max_tokens: int = 256) -> str:
        r = await self.http.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
