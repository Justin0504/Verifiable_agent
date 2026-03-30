"""Tool registry: routes claims to applicable tools and aggregates evidence.

Supports:
- Caching: avoid re-querying the same claim
- Routing memory: track which tools work best for which claim types
- Priority routing: deterministic tools first, soft evidence second
"""

from __future__ import annotations

from dataclasses import asdict

from .base import BaseTool, ToolResult
from .calculator import CalculatorTool
from .code_executor import CodeExecutorTool
from .scholar import SemanticScholarTool
from .web_search import WebSearchTool
from .wikidata import WikidataTool
from .wikipedia import WikipediaTool


class ToolRegistry:
    """Manages external tools and routes claims to applicable ones.

    Priority order:
    1. Deterministic tools (calculator, code_executor, Wikidata) — hard ground truth
    2. Non-deterministic tools (Wikipedia, Semantic Scholar, web search) — soft evidence

    Deterministic tool results override LLM-based verification.
    """

    def __init__(
        self,
        enable_web: bool = True,
        enable_wikidata: bool = True,
        enable_calculator: bool = True,
        enable_wikipedia: bool = True,
        enable_scholar: bool = True,
        enable_code_executor: bool = True,
        memory_store=None,
    ):
        self.tools: list[BaseTool] = []
        # Deterministic tools first (priority)
        if enable_calculator:
            self.tools.append(CalculatorTool())
        if enable_code_executor:
            self.tools.append(CodeExecutorTool())
        if enable_wikidata:
            self.tools.append(WikidataTool())
        # Non-deterministic tools (soft evidence)
        if enable_wikipedia:
            self.tools.append(WikipediaTool())
        if enable_scholar:
            self.tools.append(SemanticScholarTool())
        if enable_web:
            self.tools.append(WebSearchTool())
        self.memory = memory_store
        self._cache: dict = {}
        if self.memory:
            self._cache = self.memory.load_tool_cache()

    def query_all(self, claim: str) -> list[ToolResult]:
        """Run all applicable tools on a claim, with caching."""
        results: list[ToolResult] = []
        for tool in self.tools:
            if not tool.is_applicable(claim):
                continue

            # Check cache first
            cached = self._get_cached(claim, tool.name)
            if cached is not None:
                results.append(cached)
                continue

            result = tool.query(claim)
            if result.success:
                results.append(result)
                self._put_cache(claim, tool.name, result)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def get_evidence(self, claim: str) -> str:
        """Get aggregated evidence string from all applicable tools."""
        results = self.query_all(claim)
        if not results:
            return ""

        parts = []
        for r in results:
            tag = "VERIFIED" if r.confidence >= 1.0 else "REFERENCE"
            parts.append(f"[{tag}: {r.tool_name}]\n{r.evidence}")
        return "\n\n".join(parts)

    def has_deterministic_verdict(self, claim: str) -> tuple[bool, str]:
        """Check if any deterministic tool can give a hard verdict."""
        for tool in self.tools:
            if not tool.deterministic or not tool.is_applicable(claim):
                continue

            cached = self._get_cached(claim, tool.name)
            if cached is not None and cached.raw_data:
                return True, cached.evidence

            result = tool.query(claim)
            if result.success and result.raw_data:
                self._put_cache(claim, tool.name, result)
                return True, result.evidence
        return False, ""

    def log_routing_outcome(self, claim_type: str, tool_name: str, success: bool) -> None:
        """Record a tool routing decision and outcome for learning."""
        if self.memory:
            self.memory.update_tool_routing(claim_type, tool_name, success)

    def get_best_tool(self, claim_type: str) -> str | None:
        """Get the best-performing tool for a claim type based on history."""
        if not self.memory:
            return None
        routing = self.memory.load_tool_routing()
        type_routing = routing.get(claim_type, {})
        if not type_routing:
            return None
        best = max(type_routing.items(), key=lambda x: x[1].get("accuracy", 0))
        if best[1].get("uses", 0) >= 3:  # Need enough data to trust
            return best[0]
        return None

    def flush_cache(self) -> None:
        """Persist cache to disk."""
        if self.memory:
            self.memory.save_tool_cache(self._cache)

    # ── Cache helpers ──────────────────────────────────────────────

    def _get_cached(self, claim: str, tool_name: str) -> ToolResult | None:
        key = f"{tool_name}::{claim}"
        data = self._cache.get(key)
        if data is None:
            return None
        return ToolResult(
            tool_name=data.get("tool_name", tool_name),
            query=data.get("query", claim),
            evidence=data.get("evidence", ""),
            success=data.get("success", True),
            confidence=data.get("confidence", 0.5),
            raw_data=data.get("raw_data", {}),
        )

    def _put_cache(self, claim: str, tool_name: str, result: ToolResult) -> None:
        key = f"{tool_name}::{claim}"
        self._cache[key] = {
            "tool_name": result.tool_name,
            "query": result.query,
            "evidence": result.evidence,
            "success": result.success,
            "confidence": result.confidence,
            "raw_data": result.raw_data,
        }
        # Limit in-memory cache
        if len(self._cache) > 5000:
            keys = list(self._cache.keys())[:1000]
            for k in keys:
                del self._cache[k]
