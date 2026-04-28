"""Curriculum learning for GRPO training.

Scores training samples by difficulty and provides epoch-specific
data subsets: easy → all → hard-focused.

Usage:
    from src.training.curriculum import CurriculumManager

    cm = CurriculumManager(parquet_path, eval_results_path)
    for epoch in range(3):
        epoch_path = cm.get_epoch_data(epoch, total_epochs=3)
        # Use epoch_path as input to GRPO trainer
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


def score_sample_difficulty(sample: dict, error_type_acc: dict = None) -> float:
    """Score a training sample's difficulty (0=easy, 1=hard).

    Components:
    - Error type difficulty: based on model's accuracy on this type
    - Contrastive flag: contrastive pairs are harder (+0.15)
    - Source length: longer sources are harder
    - Metadata difficulty tag
    """
    score = 0.5  # Default medium

    # Error type: lower model accuracy = harder sample
    if error_type_acc:
        etype = sample.get("error_type", "")
        if etype in error_type_acc:
            # Accuracy 0% → difficulty 1.0, accuracy 100% → difficulty 0.0
            score = 1.0 - error_type_acc[etype]

    # Contrastive pairs are inherently harder (minimal edits)
    meta = sample.get("_meta", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}

    if meta.get("source") == "contrastive_pair":
        score = min(1.0, score + 0.15)

    # Hard negative replays
    if meta.get("source") == "hard_negative_replay":
        score = min(1.0, score + 0.2)

    # Explicit difficulty tag
    difficulty = sample.get("difficulty", meta.get("target_diff", ""))
    if difficulty == "hard":
        score = min(1.0, score + 0.1)
    elif difficulty == "easy":
        score = max(0.0, score - 0.1)

    return score


class CurriculumManager:
    """Manages curriculum-based data loading for GRPO training.

    Epoch schedule (for 5 epochs):
    - Epoch 1: difficulty < 0.5 (easy + medium, ~50-60% of data)
    - Epoch 2: difficulty < 0.7 (add medium-hard, ~70-80%)
    - Epoch 3: all data
    - Epoch 4: all data, 2x weight for difficulty > 0.5
    - Epoch 5: difficulty > 0.3, 2x weight for difficulty > 0.6
    """

    def __init__(self, parquet_path: str, eval_results: dict = None):
        if pq is None:
            raise ImportError("pyarrow required for CurriculumManager")

        self.parquet_path = parquet_path
        self.table = pq.read_table(parquet_path)
        self.n_samples = len(self.table)

        # Extract error type accuracies from eval results
        self.error_type_acc = {}
        if eval_results:
            for bench_results in eval_results.values():
                if isinstance(bench_results, dict):
                    etw = bench_results.get("error_type_weaknesses", {})
                    for etype, stats in etw.items():
                        self.error_type_acc[etype] = stats.get("accuracy", 0.5)

        # Score all samples
        self.difficulties = self._score_all()

    def _score_all(self) -> list[float]:
        """Score difficulty for each sample in the parquet."""
        difficulties = []
        for i in range(self.n_samples):
            sample = {}
            # Extract relevant fields from parquet row
            if "extra_info" in self.table.column_names:
                try:
                    ei = json.loads(self.table.column("extra_info")[i].as_py())
                    sample["error_type"] = ei.get("error_type", "")
                    sample["_meta"] = ei
                except (json.JSONDecodeError, TypeError):
                    pass
            if "reward_model" in self.table.column_names:
                try:
                    rm = json.loads(self.table.column("reward_model")[i].as_py())
                    gt = rm.get("ground_truth", {})
                    if isinstance(gt, dict):
                        sample["error_type"] = sample.get("error_type") or gt.get("error_type", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            difficulties.append(score_sample_difficulty(sample, self.error_type_acc))
        return difficulties

    def get_epoch_data(self, epoch: int, total_epochs: int = 5,
                       output_dir: str = None) -> str:
        """Generate epoch-specific training data based on curriculum.

        Returns path to the filtered parquet file.
        """
        output_dir = Path(output_dir or Path(self.parquet_path).parent / "curriculum")
        output_dir.mkdir(parents=True, exist_ok=True)

        progress = epoch / max(total_epochs - 1, 1)

        # Determine difficulty threshold and oversampling
        if progress < 0.25:
            # Early: easy + medium only
            max_difficulty = 0.5
            oversample_threshold = None
        elif progress < 0.5:
            # Mid-early: add medium-hard
            max_difficulty = 0.7
            oversample_threshold = None
        elif progress < 0.75:
            # Mid-late: all data
            max_difficulty = 1.0
            oversample_threshold = None
        else:
            # Late: all data, oversample hard
            max_difficulty = 1.0
            oversample_threshold = 0.6

        # Filter indices
        indices = []
        for i, d in enumerate(self.difficulties):
            if d <= max_difficulty:
                indices.append(i)
                # Oversample hard examples
                if oversample_threshold and d > oversample_threshold:
                    indices.append(i)  # 2x weight

        if not indices:
            indices = list(range(self.n_samples))

        # Create filtered table
        filtered = self.table.take(indices)
        out_path = output_dir / f"epoch_{epoch}.parquet"
        pq.write_table(filtered, out_path)

        stats = {
            "epoch": epoch,
            "total_samples": len(indices),
            "original_samples": self.n_samples,
            "max_difficulty": max_difficulty,
            "oversample_threshold": oversample_threshold,
            "difficulty_distribution": {
                "easy (<0.3)": sum(1 for i in indices if self.difficulties[i] < 0.3),
                "medium (0.3-0.6)": sum(1 for i in indices if 0.3 <= self.difficulties[i] < 0.6),
                "hard (>0.6)": sum(1 for i in indices if self.difficulties[i] >= 0.6),
            },
        }

        with open(output_dir / f"epoch_{epoch}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Curriculum epoch {epoch}: {len(indices)}/{self.n_samples} samples "
              f"(max_diff={max_difficulty}, oversample={oversample_threshold})")

        return str(out_path)

    def get_summary(self) -> dict:
        """Get difficulty distribution summary."""
        return {
            "total": self.n_samples,
            "easy (<0.3)": sum(1 for d in self.difficulties if d < 0.3),
            "medium (0.3-0.6)": sum(1 for d in self.difficulties if 0.3 <= d < 0.6),
            "hard (>0.6)": sum(1 for d in self.difficulties if d >= 0.6),
            "mean_difficulty": sum(self.difficulties) / max(len(self.difficulties), 1),
        }
