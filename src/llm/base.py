"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


class BaseLLM(ABC):
    """Abstract LLM interface. All providers must implement this."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a completion given a prompt and optional system message."""
        ...

    @abstractmethod
    def generate_with_messages(
        self, messages: list[dict], system: str | None = None
    ) -> LLMResponse:
        """Generate a completion from a list of chat messages."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
