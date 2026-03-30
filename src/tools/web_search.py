"""Web search tool for retrieving online evidence.

Uses DuckDuckGo instant answer API (no API key required) as default,
with optional support for Google/Bing APIs.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .base import BaseTool, ToolResult

DDGS_API = "https://api.duckduckgo.com/"


class WebSearchTool(BaseTool):
    """Retrieve evidence from the web via search."""

    name = "web_search"
    description = "Web search for factual evidence"
    deterministic = False

    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def query(self, claim: str) -> ToolResult:
        """Search the web for evidence related to the claim."""
        try:
            results = self._ddg_search(claim)
            if not results:
                return ToolResult(
                    tool_name=self.name,
                    query=claim,
                    evidence="No web results found.",
                    success=False,
                )

            evidence_parts = []
            for i, r in enumerate(results[: self.max_results], 1):
                evidence_parts.append(f"[Web {i}] {r['title']}\n{r['snippet']}")

            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence="\n\n".join(evidence_parts),
                success=True,
                confidence=0.7,  # Web results are not always reliable
                raw_data={"results": results},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence=f"Web search failed: {e}",
                success=False,
            )

    def _ddg_search(self, query: str) -> list[dict]:
        """Query DuckDuckGo instant answer API."""
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        })
        url = f"{DDGS_API}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VerifiableAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", ""),
                "snippet": data["Abstract"],
                "source": data.get("AbstractURL", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if "Text" in topic:
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "snippet": topic["Text"],
                    "source": topic.get("FirstURL", ""),
                })

        return results

    def is_applicable(self, claim: str) -> bool:
        """Web search is applicable to most factual claims."""
        return len(claim.split()) >= 4
