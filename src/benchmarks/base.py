"""Base class for all benchmark dataset loaders.

Each benchmark loader converts its native format into a unified
BenchmarkSample schema that the evaluation script can consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkSample:
    """Unified benchmark sample for evaluation.

    Attributes:
        id: Unique sample identifier.
        question: The probe question or claim text.
        reference_answer: Ground truth / gold answer (if available).
        gold_label: Gold verification label — "S", "C", or "N".
        claims: Pre-decomposed atomic claims (if the benchmark provides them).
        evidence: Gold evidence passages (if available).
        metadata: Extra fields (source, category, difficulty, etc.).
    """

    id: str
    question: str
    reference_answer: str = ""
    gold_label: str = ""  # "S", "C", "N" or benchmark-native label
    claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BenchmarkLoader(ABC):
    """Abstract base for benchmark dataset loaders."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def load(self, split: str = "validation", limit: int | None = None) -> list[BenchmarkSample]:
        """Load benchmark samples.

        Args:
            split: Dataset split — "train", "validation", or "test".
            limit: Max number of samples to load (None = all).

        Returns:
            List of BenchmarkSample objects.
        """
        ...

    def load_as_probes(self, split: str = "validation", limit: int | None = None) -> list[dict]:
        """Load and convert to Probe-compatible dicts for run_experiment.py.

        Returns dicts with keys: id, question, risk_type, ground_truth, metadata.
        """
        samples = self.load(split=split, limit=limit)
        probes = []
        for s in samples:
            probes.append({
                "id": s.id,
                "question": s.question,
                "risk_type": s.metadata.get("risk_type", "missing_evidence"),
                "ground_truth": s.reference_answer or None,
                "metadata": {
                    "benchmark": self.name,
                    "gold_label": s.gold_label,
                    **s.metadata,
                },
            })
        return probes

    def load_as_kb_documents(self, split: str = "train", limit: int | None = None) -> list[dict]:
        """Convert evidence from the benchmark into KB document format."""
        samples = self.load(split=split, limit=limit)
        docs = []
        for s in samples:
            for i, ev in enumerate(s.evidence):
                docs.append({
                    "id": f"{self.name}_{s.id}_ev{i}",
                    "title": f"{self.name}: {s.question[:60]}",
                    "content": ev,
                    "source": self.name,
                })
        return docs

    @staticmethod
    def _try_import_datasets():
        """Try to import HuggingFace datasets library."""
        try:
            import datasets
            return datasets
        except ImportError:
            raise ImportError(
                "HuggingFace 'datasets' library required. "
                "Install with: pip install datasets"
            )
