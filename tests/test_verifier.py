"""Tests for the Verifier module (decomposer, evidence_matcher, knowledge_base)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.schema import AtomicClaim, ClaimLabel, Probe, Response, RiskType
from src.llm.base import LLMResponse
from src.verifier.decomposer import Decomposer
from src.verifier.evidence_matcher import EvidenceMatcher
from src.verifier.knowledge_base import KnowledgeBase


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    return llm


class TestDecomposer:
    def test_decompose_json_output(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text=json.dumps([
                "Einstein was born in 1879.",
                "He was born in Ulm, Germany.",
                "He won the Nobel Prize for the photoelectric effect.",
            ]),
            model="test",
        )
        decomposer = Decomposer(mock_llm)
        claims = decomposer.decompose("When was Einstein born?", "Einstein was born in 1879 in Ulm...")

        assert len(claims) == 3
        assert all(isinstance(c, AtomicClaim) for c in claims)
        assert "1879" in claims[0].text

    def test_decompose_fallback(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text="1. Einstein was born in 1879.\n2. He was born in Ulm.",
            model="test",
        )
        decomposer = Decomposer(mock_llm)
        claims = decomposer.decompose("q", "a")
        assert len(claims) == 2


class TestEvidenceMatcher:
    def test_match_supported(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text=json.dumps({
                "label": "S",
                "confidence": 0.95,
                "evidence_snippet": "Einstein was born on March 14, 1879",
                "reasoning": "Directly stated in evidence",
            }),
            model="test",
        )
        matcher = EvidenceMatcher(mock_llm)
        claim = AtomicClaim(id="1", text="Einstein was born in 1879.")
        result = matcher.match(claim, "Einstein was born on March 14, 1879 in Ulm.")

        assert result.label == ClaimLabel.SUPPORTED
        assert result.confidence == 0.95

    def test_match_contradicted(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text='{"label": "C", "confidence": 0.9, "evidence_snippet": "photoelectric effect", "reasoning": "Wrong reason"}',
            model="test",
        )
        matcher = EvidenceMatcher(mock_llm)
        claim = AtomicClaim(id="2", text="Einstein won Nobel for relativity.")
        result = matcher.match(claim, "Einstein received Nobel for the photoelectric effect.")

        assert result.label == ClaimLabel.CONTRADICTED

    def test_match_not_mentioned(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text='{"label": "N", "confidence": 0.6, "evidence_snippet": "", "reasoning": "No info"}',
            model="test",
        )
        matcher = EvidenceMatcher(mock_llm)
        claim = AtomicClaim(id="3", text="Einstein liked pasta.")
        result = matcher.match(claim, "Einstein was a physicist.")

        assert result.label == ClaimLabel.NOT_MENTIONED


class TestKnowledgeBase:
    def test_load_jsonl(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "test.jsonl").write_text(
            '{"id": "1", "title": "Test", "content": "Hello world"}\n'
            '{"id": "2", "title": "Test2", "content": "Foo bar"}\n'
        )
        kb = KnowledgeBase(path=str(doc_dir))
        kb.load()
        assert len(kb) == 2

    def test_retrieve_returns_relevant(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "test.jsonl").write_text(
            '{"id": "1", "title": "Penicillin", "content": "Alexander Fleming discovered penicillin in 1928"}\n'
            '{"id": "2", "title": "Moon Landing", "content": "Apollo 11 landed on the Moon in 1969"}\n'
        )
        kb = KnowledgeBase(path=str(doc_dir), top_k=1)
        kb.load()
        docs = kb.retrieve("Who discovered penicillin?")
        assert len(docs) == 1
        assert "Fleming" in docs[0].content

    def test_empty_knowledge_base(self, tmp_path):
        kb = KnowledgeBase(path=str(tmp_path / "nonexistent"))
        kb.load()
        assert len(kb) == 0
        assert kb.retrieve("anything") == []
