"""Anthropic LLM provider (Claude)."""

from __future__ import annotations

import time

from anthropic import Anthropic

from .base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API wrapper."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        super().__init__(model, temperature, max_tokens)
        self.client = Anthropic()

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, system=system)

    def generate_with_messages(
        self, messages: list[dict], system: str | None = None
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            model=response.model,
        )
