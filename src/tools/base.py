"""Base class for external verification tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Standardized result from an external tool."""

    tool_name: str
    query: str
    evidence: str
    success: bool = True
    confidence: float = 1.0  # 1.0 for deterministic tools (calculator, Wikidata)
    raw_data: dict = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for external verification tools."""

    name: str = "base"
    description: str = ""
    deterministic: bool = False  # True if tool provides hard ground truth

    @abstractmethod
    def query(self, claim: str) -> ToolResult:
        """Query the tool with a claim and return evidence."""
        ...

    def is_applicable(self, claim: str) -> bool:
        """Check if this tool is relevant for the given claim.

        Override in subclasses for smarter routing.
        """
        return True
