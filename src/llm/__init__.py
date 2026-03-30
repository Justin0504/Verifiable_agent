"""LLM provider abstraction layer."""

from __future__ import annotations

from .base import BaseLLM, LLMResponse


def _get_providers() -> dict:
    """Lazy-load provider classes to avoid import errors when SDKs are missing."""
    providers = {}
    try:
        from .openai_llm import OpenAILLM
        providers["openai"] = OpenAILLM
    except ImportError:
        pass
    try:
        from .anthropic_llm import AnthropicLLM
        providers["anthropic"] = AnthropicLLM
    except ImportError:
        pass
    try:
        from .vllm_llm import VLLMOpenAI
        providers["vllm"] = VLLMOpenAI
    except ImportError:
        pass
    return providers


def create_llm(config: dict) -> BaseLLM:
    """Factory: create an LLM instance from a config dict.

    Expected keys: provider, model, temperature, max_tokens.
    Optional: api_base (for vllm / custom endpoints).
    """
    providers = _get_providers()
    provider = config["provider"]
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown or unavailable LLM provider: {provider!r}. "
            f"Available: {list(providers)}. Install the required SDK."
        )

    kwargs = {
        "model": config["model"],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }
    if "api_base" in config:
        kwargs["api_base"] = config["api_base"]
    return cls(**kwargs)


__all__ = ["BaseLLM", "LLMResponse", "create_llm"]
