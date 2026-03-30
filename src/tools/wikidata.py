"""Wikidata knowledge graph tool for structured entity queries.

Uses the Wikidata SPARQL endpoint for precise entity lookups.
Especially useful for multi-hop reasoning verification.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from .base import BaseTool, ToolResult

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"


class WikidataTool(BaseTool):
    """Query Wikidata knowledge graph for structured facts.

    Handles:
    - Entity attributes: birth date, death date, nationality, occupation
    - Relationships: spouse, children, employer, educated at
    - Multi-hop: "Who is the spouse of the person who directed X?"
    """

    name = "wikidata"
    description = "Structured knowledge graph queries"
    deterministic = True  # Wikidata facts are authoritative

    def query(self, claim: str) -> ToolResult:
        """Extract entities from the claim and look them up in Wikidata."""
        try:
            # Step 1: Find entity candidates in the claim
            entities = self._extract_entity_candidates(claim)
            if not entities:
                return ToolResult(
                    tool_name=self.name,
                    query=claim,
                    evidence="No recognizable entities found in claim.",
                    success=False,
                )

            # Step 2: Look up each entity
            all_facts: list[str] = []
            for entity_name in entities[:3]:  # Limit to 3 entities
                qid = self._search_entity(entity_name)
                if not qid:
                    continue
                facts = self._get_entity_facts(qid)
                if facts:
                    fact_lines = [f"  {k}: {v}" for k, v in facts.items()]
                    all_facts.append(
                        f"[Wikidata: {entity_name} ({qid})]\n" + "\n".join(fact_lines)
                    )

            if not all_facts:
                return ToolResult(
                    tool_name=self.name,
                    query=claim,
                    evidence="Entities found but no relevant facts retrieved.",
                    success=False,
                )

            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence="\n\n".join(all_facts),
                success=True,
                confidence=1.0,
                raw_data={"entities": entities},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence=f"Wikidata query failed: {e}",
                success=False,
            )

    def _extract_entity_candidates(self, text: str) -> list[str]:
        """Extract likely entity names (capitalized multi-word phrases)."""
        # Match capitalized words that are likely proper nouns
        # e.g. "Albert Einstein", "Sydney Opera House", "Nobel Prize"
        pattern = r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|in|for|and))*(?:\s+[A-Z][a-z]+))\b'
        matches = re.findall(pattern, text)

        # Also try single capitalized words > 3 chars (excluding sentence starts)
        singles = re.findall(r'(?<!\.\s)\b([A-Z][a-z]{3,})\b', text)

        # Deduplicate, prefer longer matches
        seen = set()
        entities = []
        for m in sorted(matches + singles, key=len, reverse=True):
            m_lower = m.lower()
            if m_lower not in seen and m_lower not in {"the", "this", "that", "when", "where", "which"}:
                seen.add(m_lower)
                entities.append(m)

        return entities[:5]

    def _search_entity(self, name: str) -> str | None:
        """Search Wikidata for an entity by name, return QID."""
        params = urllib.parse.urlencode({
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": 1,
        })
        url = f"{WIKIDATA_SEARCH}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VerifiableAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("search", [])
        if results:
            return results[0]["id"]
        return None

    def _get_entity_facts(self, qid: str) -> dict[str, str]:
        """Get key facts about an entity via SPARQL."""
        # Properties we care about for fact-checking
        properties = {
            "P569": "birth_date",
            "P570": "death_date",
            "P19": "birth_place",
            "P20": "death_place",
            "P27": "nationality",
            "P106": "occupation",
            "P166": "award",
            "P69": "educated_at",
            "P108": "employer",
            "P26": "spouse",
            "P40": "children",
            "P50": "author",
            "P57": "director",
            "P1082": "population",
            "P17": "country",
            "P36": "capital",
        }

        sparql = f"""
        SELECT ?prop ?propLabel ?val ?valLabel WHERE {{
          VALUES ?prop {{ {' '.join(f'wdt:{p}' for p in properties)} }}
          wd:{qid} ?prop ?val .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 20
        """

        params = urllib.parse.urlencode({"query": sparql, "format": "json"})
        url = f"{WIKIDATA_SPARQL}?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "VerifiableAgent/1.0",
            "Accept": "application/sparql-results+json",
        })

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return {}

        facts: dict[str, str] = {}
        prop_uri_to_name = {f"http://www.wikidata.org/prop/direct/{k}": v for k, v in properties.items()}

        for binding in data.get("results", {}).get("bindings", []):
            prop_uri = binding.get("prop", {}).get("value", "")
            prop_name = prop_uri_to_name.get(prop_uri, prop_uri)
            val = binding.get("valLabel", {}).get("value", "")
            if not val:
                val = binding.get("val", {}).get("value", "")
            if prop_name in facts:
                facts[prop_name] += f", {val}"
            else:
                facts[prop_name] = val

        return facts

    def is_applicable(self, claim: str) -> bool:
        """Check if claim mentions recognizable entities."""
        entities = self._extract_entity_candidates(claim)
        return len(entities) > 0
