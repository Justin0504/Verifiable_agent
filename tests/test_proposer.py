"""Tests for the Proposer module."""

import json
import uuid
from unittest.mock import MagicMock

import pytest

from src.data.schema import Probe, RiskType
from src.llm.base import LLMResponse
from src.proposer.proposer import Proposer


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "test-model"
    return llm


def _make_llm_response(probes: list[dict]) -> LLMResponse:
    return LLMResponse(text=json.dumps(probes), model="test")


class TestProposer:
    def test_generate_probes_parses_json(self, mock_llm):
        mock_llm.generate.return_value = _make_llm_response([
            {"question": "What was discussed in the secret meeting?", "ground_truth": "UNANSWERABLE", "reasoning": "No public record"},
            {"question": "Who attended the closed session?", "ground_truth": "UNANSWERABLE", "reasoning": "Classified"},
        ])
        proposer = Proposer(mock_llm, seed=42)
        probes = proposer.generate_probes(RiskType.MISSING_EVIDENCE, n=2)

        assert len(probes) == 2
        assert all(isinstance(p, Probe) for p in probes)
        assert probes[0].risk_type == RiskType.MISSING_EVIDENCE
        assert probes[0].question == "What was discussed in the secret meeting?"

    def test_generate_probes_fallback_parse(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            text="1. What is the secret formula?\n2. Who invented time travel?",
            model="test",
        )
        proposer = Proposer(mock_llm)
        probes = proposer.generate_probes(RiskType.UNANSWERABLE, n=2)

        assert len(probes) == 2
        assert all("?" in p.question for p in probes)

    def test_generate_all_covers_all_risk_types(self, mock_llm):
        mock_llm.generate.return_value = _make_llm_response([
            {"question": "Test question?", "ground_truth": "test"},
        ])
        proposer = Proposer(mock_llm, seed=42)
        probes = proposer.generate_all(n_per_type=1)

        risk_types_seen = {p.risk_type for p in probes}
        assert risk_types_seen == set(RiskType)

    def test_update_memory(self, mock_llm):
        proposer = Proposer(mock_llm)
        assert len(proposer.memory) == 0
        proposer.update_memory([{"pattern": "test", "failure_type": "boundary"}])
        assert len(proposer.memory) == 1

    def test_empty_response_returns_empty(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(text="", model="test")
        proposer = Proposer(mock_llm)
        probes = proposer.generate_probes(RiskType.MULTI_HOP, n=5)
        assert probes == []
