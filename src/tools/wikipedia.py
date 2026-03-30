"""Wikipedia API tool for retrieving encyclopedic evidence.

Uses the Wikipedia REST API (no API key required).
Good for entity summaries, event descriptions, and general facts.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .base import BaseTool, ToolResult

WIKI_API = "https://en.wikipedia.org/api/rest_v1"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"


class WikipediaTool(BaseTool):
    """Retrieve evidence from Wikipedia articles."""

    name = "wikipedia"
    description = "Wikipedia article summaries and facts"
    deterministic = False  # Wikipedia is reliable but not perfectly authoritative

    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences

    def query(self, claim: str) -> ToolResult:
        """Search Wikipedia for articles related to the claim."""
        try:
            # Step 1: Search for relevant articles
            titles = self._search(claim)
            if not titles:
                return ToolResult(
                    tool_name=self.name, query=claim,
                    evidence="No Wikipedia articles found.", success=False,
                )

            # Step 2: Get summaries for top results
            evidence_parts = []
            for title in titles[:2]:
                summary = self._get_summary(title)
                if summary:
                    evidence_parts.append(f"[Wikipedia: {title}]\n{summary}")

            if not evidence_parts:
                return ToolResult(
                    tool_name=self.name, query=claim,
                    evidence="Articles found but summaries unavailable.", success=False,
                )

            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence="\n\n".join(evidence_parts),
                success=True,
                confidence=0.85,
                raw_data={"titles": titles},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, query=claim,
                evidence=f"Wikipedia query failed: {e}", success=False,
            )

    def _search(self, query: str, limit: int = 3) -> list[str]:
        """Search Wikipedia for article titles."""
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        })
        url = f"{WIKI_SEARCH_API}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VerifiableAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return [r["title"] for r in data.get("query", {}).get("search", [])]

    def _get_summary(self, title: str) -> str | None:
        """Get the summary of a Wikipedia article."""
        encoded = urllib.parse.quote(title)
        url = f"{WIKI_API}/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "VerifiableAgent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            extract = data.get("extract", "")
            if extract:
                sentences = extract.split(". ")
                return ". ".join(sentences[:self.max_sentences]) + "."
            return None
        except Exception:
            return None

    def is_applicable(self, claim: str) -> bool:
        return len(claim.split()) >= 4
