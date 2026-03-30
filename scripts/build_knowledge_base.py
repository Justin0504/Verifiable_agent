"""Build and validate the knowledge base index.

Usage:
    python scripts/build_knowledge_base.py [--path knowledge_base/documents]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verifier.knowledge_base import KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Build knowledge base index")
    parser.add_argument("--path", default="knowledge_base/documents", help="Path to documents dir")
    args = parser.parse_args()

    kb = KnowledgeBase(path=args.path)
    kb.load()
    print(f"Loaded {len(kb)} documents from {args.path}")

    # Smoke test retrieval
    test_queries = [
        "Who discovered penicillin?",
        "When was the Berlin Wall built?",
        "What is the population of Shanghai?",
    ]
    for query in test_queries:
        docs = kb.retrieve(query, top_k=3)
        print(f"\nQuery: {query}")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc.title} ({doc.source})")

    print("\nKnowledge base built successfully.")


if __name__ == "__main__":
    main()
