"""LLM provider abstraction layer."""

from __future__ import annotations

from .anthropic_llm import AnthropicLLM
from .base import BaseLLM, LLMResponse
from .openai_llm import OpenAILLM
from .vllm_llm import VLLMOpenAI

_PROVIDERS = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "vllm": VLLMOpenAI,
}


def create_llm(config: dict) -> BaseLLM:
    """Factory: create an LLM instance from a config dict.

    Expected keys: provider, model, temperature, max_tokens.
    Optional: api_base (for vllm / custom endpoints).
    """
    provider = config["provider"]
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Choose from {list(_PROVIDERS)}")

    kwargs = {
        "model": config["model"],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }
    if "api_base" in config:
        kwargs["api_base"] = config["api_base"]
    return cls(**kwargs)


__all__ = ["BaseLLM", "LLMResponse", "OpenAILLM", "AnthropicLLM", "VLLMOpenAI", "create_llm"]
