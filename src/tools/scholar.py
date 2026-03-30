"""Semantic Scholar API tool for verifying academic/scientific claims.

Uses the free Semantic Scholar API (no key required for basic access).
Good for verifying claims about papers, authors, citations, and findings.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .base import BaseTool, ToolResult

S2_API = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarTool(BaseTool):
    """Verify academic claims via Semantic Scholar."""

    name = "semantic_scholar"
    description = "Academic paper search and citation verification"
    deterministic = False  # Metadata is authoritative, but relevance matching is soft

    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def query(self, claim: str) -> ToolResult:
        """Search for papers related to the claim."""
        try:
            papers = self._search_papers(claim)
            if not papers:
                return ToolResult(
                    tool_name=self.name, query=claim,
                    evidence="No relevant papers found.", success=False,
                )

            evidence_parts = []
            for p in papers[:self.max_results]:
                parts = [f"[Paper: {p.get('title', 'Unknown')}]"]
                if p.get("authors"):
                    author_names = [a.get("name", "") for a in p["authors"][:5]]
                    parts.append(f"  Authors: {', '.join(author_names)}")
                if p.get("year"):
                    parts.append(f"  Year: {p['year']}")
                if p.get("citationCount") is not None:
                    parts.append(f"  Citations: {p['citationCount']}")
                if p.get("tldr") and p["tldr"].get("text"):
                    parts.append(f"  Summary: {p['tldr']['text']}")
                elif p.get("abstract"):
                    abstract = p["abstract"][:300]
                    parts.append(f"  Abstract: {abstract}...")
                if p.get("venue"):
                    parts.append(f"  Venue: {p['venue']}")
                evidence_parts.append("\n".join(parts))

            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence="\n\n".join(evidence_parts),
                success=True,
                confidence=0.8,
                raw_data={"papers": [p.get("title") for p in papers]},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, query=claim,
                evidence=f"Semantic Scholar query failed: {e}", success=False,
            )

    def _search_papers(self, query: str, limit: int = 5) -> list[dict]:
        """Search Semantic Scholar for papers."""
        params = urllib.parse.urlencode({
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,venue,tldr",
        })
        url = f"{S2_API}/paper/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VerifiableAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get("data", [])

    def is_applicable(self, claim: str) -> bool:
        """Check if claim is likely academic/scientific."""
        academic_keywords = [
            "study", "research", "paper", "published", "journal", "found that",
            "discovered", "experiment", "hypothesis", "theory", "proved",
            "Nobel", "prize", "citation", "peer-reviewed", "et al",
            "CRISPR", "genome", "quantum", "neural", "algorithm",
        ]
        claim_lower = claim.lower()
        return any(kw in claim_lower for kw in academic_keywords)
