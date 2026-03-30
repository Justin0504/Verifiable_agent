"""Persistent memory store for cross-experiment evolution state.

Complete memory system covering:
1. Proposer strategies
2. Verifier few-shot corrections (matcher + decomposer)
3. Calibration history (accuracy trend across epochs)
4. Tool routing decisions (which tool worked for which claim type)
5. Tool result cache (avoid re-querying the same claim)
6. Verification traces (full claim → evidence → CoT → verdict chain)
7. KB provenance (what was added, when, from which claim)
8. Raw failure history
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .failure_extractor import InformativeFailure

DEFAULT_MEMORY_DIR = "memory"


class MemoryStore:
    """Persistent JSON-backed memory for the evolution loop."""

    def __init__(self, path: str = DEFAULT_MEMORY_DIR):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Existing memory files
        self._proposer_file = self.path / "proposer_memory.json"
        self._verifier_matcher_file = self.path / "verifier_matcher_fewshots.json"
        self._verifier_decomposer_file = self.path / "verifier_decomposer_fewshots.json"
        self._calibration_file = self.path / "calibration_correction.json"
        self._failures_file = self.path / "failures_history.jsonl"

        # New memory files
        self._calibration_history_file = self.path / "calibration_history.jsonl"
        self._tool_routing_file = self.path / "tool_routing.json"
        self._tool_cache_file = self.path / "tool_cache.json"
        self._verification_traces_file = self.path / "verification_traces.jsonl"
        self._kb_provenance_file = self.path / "kb_provenance.jsonl"

    # ══════════════════════════════════════════════════════════════════
    # 1. Proposer Memory
    # ══════════════════════════════════════════════════════════════════

    def load_proposer_memory(self) -> list[dict]:
        return self._load_json(self._proposer_file, default=[])

    def save_proposer_memory(self, memory: list[dict]) -> None:
        self._save_json(self._proposer_file, memory)

    # ══════════════════════════════════════════════════════════════════
    # 2. Verifier Few-Shot Corrections
    # ══════════════════════════════════════════════════════════════════

    def load_matcher_few_shots(self) -> list[dict]:
        return self._load_json(self._verifier_matcher_file, default=[])

    def save_matcher_few_shots(self, examples: list[dict]) -> None:
        self._save_json(self._verifier_matcher_file, examples)

    def load_decomposer_few_shots(self) -> list[dict]:
        return self._load_json(self._verifier_decomposer_file, default=[])

    def save_decomposer_few_shots(self, examples: list[dict]) -> None:
        self._save_json(self._verifier_decomposer_file, examples)

    # ══════════════════════════════════════════════════════════════════
    # 3. Calibration (current correction + historical accuracy trend)
    # ══════════════════════════════════════════════════════════════════

    def load_calibration_correction(self) -> str:
        data = self._load_json(self._calibration_file, default={})
        return data.get("correction", "")

    def save_calibration_correction(self, correction: str) -> None:
        self._save_json(self._calibration_file, {"correction": correction})

    def append_calibration_history(self, epoch: int, experiment_id: str, results: dict) -> None:
        """Log calibration accuracy after each epoch for trend analysis."""
        record = {
            "epoch": epoch,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "accuracy": results.get("accuracy", 0),
            "per_label": results.get("per_label", {}),
            "biases": results.get("biases", []),
        }
        self._append_jsonl(self._calibration_history_file, record)

    def load_calibration_history(self) -> list[dict]:
        return self._load_jsonl(self._calibration_history_file)

    # ══════════════════════════════════════════════════════════════════
    # 4. Tool Routing Decisions
    # ══════════════════════════════════════════════════════════════════

    def load_tool_routing(self) -> dict:
        """Load learned tool routing preferences.

        Structure: {claim_type: {tool_name: {uses: N, successes: N, accuracy: float}}}
        """
        return self._load_json(self._tool_routing_file, default={})

    def save_tool_routing(self, routing: dict) -> None:
        self._save_json(self._tool_routing_file, routing)

    def update_tool_routing(self, claim_type: str, tool_name: str, success: bool) -> None:
        """Record a tool routing decision and its outcome."""
        routing = self.load_tool_routing()
        if claim_type not in routing:
            routing[claim_type] = {}
        if tool_name not in routing[claim_type]:
            routing[claim_type][tool_name] = {"uses": 0, "successes": 0, "accuracy": 0.0}

        entry = routing[claim_type][tool_name]
        entry["uses"] += 1
        if success:
            entry["successes"] += 1
        entry["accuracy"] = entry["successes"] / entry["uses"]

        self.save_tool_routing(routing)

    # ══════════════════════════════════════════════════════════════════
    # 5. Tool Result Cache
    # ══════════════════════════════════════════════════════════════════

    def load_tool_cache(self) -> dict:
        """Load cached tool results. Key: claim text hash, Value: ToolResult dict."""
        return self._load_json(self._tool_cache_file, default={})

    def save_tool_cache(self, cache: dict) -> None:
        self._save_json(self._tool_cache_file, cache)

    def get_cached_tool_result(self, claim: str, tool_name: str) -> dict | None:
        """Look up a cached tool result for a claim."""
        cache = self.load_tool_cache()
        key = f"{tool_name}::{claim}"
        return cache.get(key)

    def cache_tool_result(self, claim: str, tool_name: str, result: dict) -> None:
        """Cache a tool result to avoid re-querying."""
        cache = self.load_tool_cache()
        key = f"{tool_name}::{claim}"
        result["cached_at"] = datetime.now().isoformat()
        cache[key] = result
        # Limit cache size
        if len(cache) > 5000:
            oldest_keys = sorted(cache, key=lambda k: cache[k].get("cached_at", ""))[:1000]
            for k in oldest_keys:
                del cache[k]
        self.save_tool_cache(cache)

    # ══════════════════════════════════════════════════════════════════
    # 6. Verification Traces (full reasoning chain)
    # ══════════════════════════════════════════════════════════════════

    def append_verification_trace(self, trace: dict) -> None:
        """Log full verification trace for a single claim.

        Expected fields:
        - claim_id, claim_text, probe_id, probe_question
        - evidence_sources: [{source: "kb"|"calculator"|"wikidata"|"web", content: ...}]
        - cot_reasoning: the chain-of-thought steps from EvidenceMatcher
        - predicted_label, confidence
        - tool_used, tool_was_deterministic
        - epoch, experiment_id, model_name
        """
        trace["timestamp"] = datetime.now().isoformat()
        self._append_jsonl(self._verification_traces_file, trace)

    def load_verification_traces(self, last_n: int = 100) -> list[dict]:
        """Load recent verification traces."""
        all_traces = self._load_jsonl(self._verification_traces_file)
        return all_traces[-last_n:]

    # ══════════════════════════════════════════════════════════════════
    # 7. KB Provenance (what was auto-added, when, why)
    # ══════════════════════════════════════════════════════════════════

    def append_kb_provenance(
        self,
        doc_id: str,
        claim_text: str,
        evidence_text: str,
        source_probe_id: str,
        epoch: int,
        experiment_id: str,
        confidence: float,
    ) -> None:
        """Log when and why a new document was added to the KB."""
        record = {
            "doc_id": doc_id,
            "claim_text": claim_text,
            "evidence_text": evidence_text[:500],
            "source_probe_id": source_probe_id,
            "epoch": epoch,
            "experiment_id": experiment_id,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self._append_jsonl(self._kb_provenance_file, record)

    def load_kb_provenance(self) -> list[dict]:
        return self._load_jsonl(self._kb_provenance_file)

    # ══════════════════════════════════════════════════════════════════
    # 8. Raw Failure History
    # ══════════════════════════════════════════════════════════════════

    def append_failures(self, failures: list[InformativeFailure], epoch: int, experiment_id: str) -> None:
        with open(self._failures_file, "a") as f:
            for failure in failures:
                record = asdict(failure)
                record["epoch"] = epoch
                record["experiment_id"] = experiment_id
                record["timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(record, default=str) + "\n")

    # ══════════════════════════════════════════════════════════════════
    # Stats & Reset
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        proposer = self.load_proposer_memory()
        matcher = self.load_matcher_few_shots()
        decomposer = self.load_decomposer_few_shots()
        routing = self.load_tool_routing()
        cache = self.load_tool_cache()
        cal_history = self.load_calibration_history()

        n_failures = len(self._load_jsonl(self._failures_file))
        n_traces = len(self._load_jsonl(self._verification_traces_file))
        n_kb_adds = len(self._load_jsonl(self._kb_provenance_file))

        return {
            "proposer_strategies": len(proposer),
            "matcher_few_shots": len(matcher),
            "decomposer_few_shots": len(decomposer),
            "calibration_epochs": len(cal_history),
            "latest_accuracy": cal_history[-1]["accuracy"] if cal_history else None,
            "tool_routing_entries": sum(len(v) for v in routing.values()),
            "tool_cache_size": len(cache),
            "verification_traces": n_traces,
            "kb_auto_additions": n_kb_adds,
            "total_failures_logged": n_failures,
        }

    def reset(self) -> None:
        """Clear all persisted memory."""
        for f in [
            self._proposer_file, self._verifier_matcher_file,
            self._verifier_decomposer_file, self._calibration_file,
            self._failures_file, self._calibration_history_file,
            self._tool_routing_file, self._tool_cache_file,
            self._verification_traces_file, self._kb_provenance_file,
        ]:
            if f.exists():
                f.unlink()

    # ══════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ══════════════════════════════════════════════════════════════════

    def _load_json(self, filepath: Path, default=None):
        if not filepath.exists():
            return default if default is not None else {}
        with open(filepath) as f:
            return json.load(f)

    def _save_json(self, filepath: Path, data) -> None:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _append_jsonl(self, filepath: Path, record: dict) -> None:
        with open(filepath, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _load_jsonl(self, filepath: Path) -> list[dict]:
        if not filepath.exists():
            return []
        records = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
