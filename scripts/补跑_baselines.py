"""补跑 v9 缺失的 baselines: cove, selfcheck_gpt, retrieve_nli.

复用与 run_fair_comparison.py v9 完全相同的 test set (seed=42, cal_ratio=0.2).
结果合并到 fair_comparison_truthfulqa_v9/comparison.json.

Usage:
    python scripts/补跑_baselines.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from src.baselines.base import BaseBaseline
from src.baselines.cove import CoVeBaseline
from src.baselines.selfcheck import SelfCheckGPTBaseline
from src.baselines.retrieve_nli import RetrieveNLIBaseline
from src.benchmarks import TruthfulQALoader
from src.llm.openai_llm import OpenAILLM

OUTPUT_DIR = Path("results/fair_comparison_truthfulqa_v9")
COMPARISON_FILE = OUTPUT_DIR / "comparison.json"

BASELINES_TO_RUN = {
    "cove": lambda llm: CoVeBaseline(llm, n_questions=2),
    "selfcheck_gpt": lambda llm: SelfCheckGPTBaseline(llm, n_samples=3),
    "retrieve_nli": lambda llm: RetrieveNLIBaseline(llm),
}


def main():
    # Reproduce the exact same test set as v9
    loader = TruthfulQALoader()
    try:
        all_samples = loader.load(split="validation", limit=200)
    except Exception as e:
        print(f"HuggingFace failed ({e}), manual fallback...")
        all_samples = loader.load_manual_sample(limit=50)

    all_samples = [s for s in all_samples if s.gold_label in ("S", "C", "N")]
    random.Random(42).shuffle(all_samples)

    n_cal = max(10, int(len(all_samples) * 0.2))
    test_samples = all_samples[n_cal:]
    test_dist = Counter(s.gold_label for s in test_samples)
    print(f"Test set: {len(test_samples)} samples {dict(test_dist)}")

    # Load existing results
    with open(COMPARISON_FILE) as f:
        comparison = json.load(f)

    existing = set(comparison["metrics"].keys())
    to_run = {k: v for k, v in BASELINES_TO_RUN.items() if k not in existing}

    if not to_run:
        print("All baselines already completed!")
        return

    print(f"Need to run: {list(to_run.keys())}")
    print()

    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.2)

    for bl_name, bl_factory in to_run.items():
        print(f"  {bl_name}...")
        bl = bl_factory(llm)
        t0 = time.time()

        bl_results = []
        for i, s in enumerate(test_samples):
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(test_samples)}]...")
            r = bl.verify_sample(s)
            bl_results.append(r)

        bm = BaseBaseline.compute_metrics(bl_results)
        elapsed = time.time() - t0
        print(f"    {bm['accuracy']:.1%} (F1={bm['macro_f1']:.3f}, {elapsed:.0f}s)")

        # Merge into comparison
        comparison["metrics"][bl_name] = {
            "accuracy": bm["accuracy"],
            "macro_f1": bm["macro_f1"],
            "confusion_matrix": bm.get("confusion_matrix", {}),
        }

        # Save after each baseline (in case of crash)
        with open(COMPARISON_FILE, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"    Saved to {COMPARISON_FILE}")

    # Final ranking
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: truthfulqa | {len(test_samples)} test samples | v9")
    print("=" * 70)

    ranked = sorted(comparison["metrics"].items(), key=lambda x: -x[1]["accuracy"])
    print(f"\n{'Method':<25} {'Accuracy':>10} {'F1':>8} {'Rank':>6}")
    print("-" * 53)
    for rank, (name, m) in enumerate(ranked, 1):
        marker = " ***" if name == "verifier_agent" else ""
        print(f"{name:<25} {m['accuracy']:>9.1%} {m['macro_f1']:>8.3f} {rank:>5}{marker}")

    # Update ranking in comparison
    comparison["ranking"] = [(n, m["accuracy"]) for n, m in ranked]
    v_rank = next(i for i, (n, _) in enumerate(ranked, 1) if n == "verifier_agent")
    comparison["verifier_rank"] = v_rank

    with open(COMPARISON_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nDone! Results saved to {COMPARISON_FILE}")


if __name__ == "__main__":
    main()
