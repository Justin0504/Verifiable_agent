"""Knowledge base for evidence retrieval.

Supports TF-IDF and embedding-based retrieval from pre-built document collections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """A single knowledge base document."""

    id: str
    title: str
    content: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class KnowledgeBase:
    """Retrieve evidence from a pre-built document collection."""

    def __init__(self, path: str, retrieval_method: str = "tfidf", top_k: int = 5):
        self.path = Path(path)
        self.retrieval_method = retrieval_method
        self.top_k = top_k
        self.documents: list[Document] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

    def load(self) -> None:
        """Load documents from the knowledge base directory."""
        self.documents = []
        if not self.path.exists():
            return

        for filepath in sorted(self.path.glob("*.jsonl")):
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.documents.append(
                        Document(
                            id=data.get("id", ""),
                            title=data.get("title", ""),
                            content=data.get("content", ""),
                            source=data.get("source", filepath.stem),
                            metadata=data.get("metadata", {}),
                        )
                    )

        for filepath in sorted(self.path.glob("*.json")):
            with open(filepath) as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    self.documents.append(
                        Document(
                            id=item.get("id", ""),
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            source=item.get("source", filepath.stem),
                            metadata=item.get("metadata", {}),
                        )
                    )

        if self.documents and self.retrieval_method == "tfidf":
            self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        """Build TF-IDF index over document contents."""
        corpus = [f"{doc.title}. {doc.content}" for doc in self.documents]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        k = top_k or self.top_k
        if not self.documents:
            return []

        if self.retrieval_method == "tfidf":
            return self._retrieve_tfidf(query, k)
        return self.documents[:k]  # Fallback: return first k

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[Document]:
        """TF-IDF based retrieval."""
        if self._vectorizer is None or self._tfidf_matrix is None:
            return self.documents[:top_k]

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

    def get_evidence_text(self, query: str, top_k: int | None = None) -> str:
        """Retrieve documents and format as a single evidence string."""
        docs = self.retrieve(query, top_k)
        if not docs:
            return "No evidence found in the knowledge base."
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Evidence {i}] {doc.title}\n{doc.content}")
        return "\n\n".join(parts)

    def add_verified_evidence(self, claim: str, evidence: str, source: str = "verified") -> None:
        """Add a verified claim-evidence pair back to the KB.

        Called after verification: high-confidence Supported claims with their
        evidence become new KB entries, so the knowledge base grows over time.
        """
        doc_id = f"auto_{len(self.documents):04d}"
        doc = Document(
            id=doc_id,
            title=claim[:100],
            content=evidence,
            source=source,
            metadata={"auto_added": True},
        )
        self.documents.append(doc)

        # Persist to disk
        auto_file = self.path / "auto_verified.jsonl"
        with open(auto_file, "a") as f:
            f.write(json.dumps({
                "id": doc_id,
                "title": doc.title,
                "content": doc.content,
                "source": source,
                "metadata": doc.metadata,
            }) + "\n")

        # Rebuild index to include the new document
        if self.retrieval_method == "tfidf":
            self._build_tfidf_index()

    def __len__(self) -> int:
        return len(self.documents)
