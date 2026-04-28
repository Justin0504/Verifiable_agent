"""Process FEVER + TruthfulQA data into veRL-compatible parquet format.

Creates train/val/test splits with the schema expected by veRL's RLHFDataset:
  - prompt: list[dict] (chat messages)
  - data_source: str
  - reward_model: {"ground_truth": {"target": label}}
  - extra_info: {"index": int, "split": str}

Usage:
    python process_fact_verification.py --output-dir ./data/fact_verification
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a fact verification expert. Given a claim and evidence, "
    "classify the claim and explain your reasoning.\n\n"
    "Labels:\n"
    "- S (Supported): The evidence supports the claim\n"
    "- C (Contradicted): The evidence contradicts the claim\n"
    "- N (Not Enough Info): The evidence is insufficient\n\n"
    'Respond with JSON only: {"label": "S/C/N", "confidence": 0.0-1.0, "reasoning": "..."}'
)

USER_TEMPLATE = "Claim: {claim}\n\nEvidence: {evidence}\n\nClassify this claim. Respond with JSON only."


def load_fever_samples(limit: int = 5000) -> list[dict]:
    """Load balanced FEVER samples."""
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")
    label_map = {"SUPPORTS": "S", "REFUTES": "C", "NOT ENOUGH INFO": "N"}
    by_label: dict[str, list] = {"S": [], "C": [], "N": []}

    for row in ds:
        label = label_map.get(row["label"])
        if label is None:
            continue
        ev_list = row.get("evidence", [])
        evidence = " ".join(e[2] if len(e) > 2 else str(e) for e in ev_list) if ev_list else ""
        if not evidence.strip():
            continue
        by_label[label].append({"claim": row["claim"], "evidence": evidence, "gold": label})

    per_class = limit // 3
    rng = random.Random(42)
    samples = []
    for label in ["S", "C", "N"]:
        pool = by_label[label]
        rng.shuffle(pool)
        samples.extend(pool[:per_class])
    return samples


def load_truthfulqa_samples(limit: int = 600) -> list[dict]:
    """Load balanced TruthfulQA samples."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rng = random.Random(42)
    all_rows = list(ds)
    rng.shuffle(all_rows)

    per_class = limit // 3
    s_samples, c_samples, n_samples = [], [], []

    for row in all_rows:
        question = row["question"]
        correct = row.get("correct_answers", [])
        incorrect = row.get("incorrect_answers", [])

        if len(s_samples) < per_class and correct:
            s_samples.append({"claim": question, "evidence": correct[0], "gold": "S"})
        if len(c_samples) < per_class and incorrect:
            c_samples.append({"claim": question, "evidence": incorrect[0], "gold": "C"})
        if len(s_samples) >= per_class and len(c_samples) >= per_class:
            break

    for i in range(min(per_class, len(all_rows))):
        j = (i + len(all_rows) // 2) % len(all_rows)
        n_samples.append({"claim": all_rows[i]["question"], "evidence": all_rows[j]["best_answer"], "gold": "N"})
        if len(n_samples) >= per_class:
            break

    return s_samples + c_samples + n_samples


def format_for_verl(samples: list[dict], data_source: str = "fact_verification") -> pd.DataFrame:
    """Convert samples to veRL-compatible DataFrame."""
    rows = []
    for i, s in enumerate(samples):
        user_content = USER_TEMPLATE.format(claim=s["claim"], evidence=s["evidence"])
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        rows.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": "fact_verification",
            "reward_model": {
                "ground_truth": {"target": s["gold"]},  # S, C, or N
            },
            "extra_info": {
                "index": i,
                "split": "train",
            },
            "metadata": {
                "claim": s["claim"],
                "evidence": s["evidence"][:200],
                "source": data_source,
            },
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data/fact_verification")
    parser.add_argument("--fever-limit", type=int, default=5000)
    parser.add_argument("--tqa-limit", type=int, default=600)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FEVER...")
    fever = load_fever_samples(args.fever_limit)
    print(f"  {len(fever)} samples, dist: {dict(Counter(s['gold'] for s in fever))}")

    print("Loading TruthfulQA...")
    tqa = load_truthfulqa_samples(args.tqa_limit)
    print(f"  {len(tqa)} samples, dist: {dict(Counter(s['gold'] for s in tqa))}")

    all_samples = fever + tqa
    rng = random.Random(42)
    rng.shuffle(all_samples)

    # Split
    n = len(all_samples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_test - n_val

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"\nSplit: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Convert to parquet
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        df = format_for_verl(samples, data_source="fact_verification")
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path)
        dist = Counter(s["gold"] for s in samples)
        print(f"  {name}: {len(samples)} samples → {path} | dist={dict(dist)}")

    # Also save test as separate dir (for veRL validation_data_dir)
    test_dir = output_dir / "test"
    test_dir.mkdir(exist_ok=True)
    df_test = format_for_verl(test_samples, data_source="fact_verification")
    df_test.to_parquet(test_dir / "test.parquet")

    print(f"\nDone! Data saved to {output_dir}")
    print(f"  Train: {output_dir / 'train.parquet'}")
    print(f"  Val:   {output_dir / 'val.parquet'}")
    print(f"  Test:  {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
