"""OpenAI LLM provider (GPT-4o, GPT-4o-mini, etc.)."""

from __future__ import annotations

import time

from openai import OpenAI

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper with automatic retry on rate limits."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_base: str | None = None,
        max_retries: int = 10,
    ):
        super().__init__(model, temperature, max_tokens)
        kwargs = {"timeout": 120.0}  # 2 min timeout per request
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.max_retries = max_retries

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

        # Proactive rate-limit avoidance: small delay between calls
        if hasattr(self, '_last_call_time'):
            elapsed = time.perf_counter() - self._last_call_time
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

        last_err = None
        for attempt in range(self.max_retries):
            try:
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
                self._last_call_time = time.perf_counter()
                return LLMResponse(
                    text=choice.message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency,
                    model=response.model,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                # Retry on ALL errors (network, timeout, rate limit, etc.)
                # Only non-retryable: KeyboardInterrupt (caught above)
                last_err = e
                wait = min(2 ** attempt * 2, 60)
                err_str = str(e).lower()
                reason = "Rate limit" if "rate" in err_str or "429" in str(e) else "Transient error"
                print(f"  [{reason}: {type(e).__name__}] Waiting {wait}s (attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_err}")
