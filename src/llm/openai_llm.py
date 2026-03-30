"""OpenAI LLM provider (GPT-4o, GPT-4o-mini, etc.)."""

from __future__ import annotations

import time

from openai import OpenAI

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_base: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens)
        kwargs = {}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)

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
            model=response.model,
        )
