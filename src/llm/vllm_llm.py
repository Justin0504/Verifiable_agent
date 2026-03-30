"""vLLM provider for open-source models (Llama-3, Mistral, etc.).

Uses the OpenAI-compatible API that vLLM exposes, so this is a thin
wrapper around the OpenAI client pointed at a local endpoint.
"""

from __future__ import annotations

import time

from openai import OpenAI

from .base import BaseLLM, LLMResponse


class VLLMOpenAI(BaseLLM):
    """vLLM via its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_base: str = "http://localhost:8000/v1",
    ):
        super().__init__(model, temperature, max_tokens)
        self.client = OpenAI(base_url=api_base, api_key="EMPTY")

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_messages(messages)

    def generate_with_messages(
        self, messages: list[dict], system: str | None = None
    ) -> LLMResponse:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model=self.model,
        )
