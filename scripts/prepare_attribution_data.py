"""Prepare fact attribution data for SEVA training and evaluation.

Downloads and unifies:
  - ANLI (NLI → attribution, for SFT training)
  - ClearFacts (clean evaluation)
  - GrayFacts (ambiguous evaluation)
  - LLM-AggreFact (11 sub-benchmarks, cross-domain evaluation)
  - SciFact, HoVer, CoverBench (depth evaluation)

Outputs:
  data/attribution/sft_train.jsonl   (SFT training data, chat format)
  data/attribution/grpo_train.parquet (GRPO training data, veRL format)
  data/attribution/clearfacts.jsonl   (evaluation)
  data/attribution/grayfacts.jsonl    (evaluation)
  data/attribution/aggrefact_*.jsonl  (per-sub-benchmark evaluation)
  data/attribution/scifact.jsonl      (evaluation)
  data/attribution/hover.jsonl        (evaluation)
  data/attribution/coverbench.jsonl   (evaluation)

Usage:
    python scripts/prepare_attribution_data.py
    python scripts/prepare_attribution_data.py --sft-only  # skip eval sets
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

# Will be imported if available
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: `datasets` not installed. Run: pip install datasets")


# ============================================================
# Constants
# ============================================================
OUTPUT_DIR = Path("/Users/justin/Verifiable_agent/data/attribution")

SYSTEM_PROMPT = (
    "You are a fact attribution verifier. Given a claim and a source document, "
    "determine whether the claim is attributable to (supported by) the source.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"label": "Attributable" or "Not Attributable", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "brief explanation"}'
)

SYSTEM_PROMPT_WITH_RULES = (
    "You are a fact attribution verifier. Given a claim and a source document, "
    "determine whether the claim is attributable to (supported by) the source.\n\n"
    "You have access to the following verification rules from the ReasoningBank. "
    "You MUST cite which rule(s) you applied in your response.\n\n"
    "{rules}\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"label": "Attributable" or "Not Attributable", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "brief explanation", '
    '"rules_cited": ["R1", ...]}'
)

USER_TEMPLATE = (
    "Claim: {claim}\n\n"
    "Source: {source}\n\n"
    "Is this claim attributable to the source? Respond with JSON only."
)


# ============================================================
# Label mapping
# ============================================================
def nli_to_attribution(label) -> str:
    """Map NLI labels to binary attribution."""
    if isinstance(label, int):
        # ANLI: 0=entailment, 1=neutral, 2=contradiction
        return "Attributable" if label == 0 else "Not Attributable"
    label = str(label).lower().strip()
    if label in ("entailment", "entailed", "supported", "s", "1", "true",
                 "attributable", "consistent"):
        return "Attributable"
    return "Not Attributable"


def make_sft_sample(claim: str, source: str, label: str, reasoning: str = "") -> dict:
    """Create a chat-format SFT sample."""
    if not reasoning:
        if label == "Attributable":
            reasoning = "The claim is directly supported by the information in the source document."
        else:
            reasoning = "The claim contains information that is not supported by or contradicts the source document."

    confidence = 0.9 if reasoning else 0.7

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(claim=claim, source=source)},
        {"role": "assistant", "content": json.dumps({
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning
        })}
    ]
    return {
        "messages": messages,
        "metadata": {"gold_label": label, "source": "anli"}
    }


def make_grpo_sample(claim: str, source: str, label: str,
                     data_source: str = "attribution") -> dict:
    """Create a veRL-format GRPO sample."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{USER_TEMPLATE.format(claim=claim, source=source)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "fact_attribution",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": label}
        },
        "extra_info": {}
    }


def make_eval_sample(claim: str, source: str, label: str,
                     benchmark: str = "", subset: str = "",
                     metadata: dict = None) -> dict:
    """Create a unified evaluation sample."""
    return {
        "claim": claim,
        "source": source,
        "gold_label": label,
        "benchmark": benchmark,
        "subset": subset,
        "metadata": metadata or {}
    }


# ============================================================
# Data loaders
# ============================================================
def load_anli(max_samples: int = 60000) -> list:
    """Load ANLI and convert to attribution format."""
    print("Loading ANLI...")
    ds = load_dataset("facebook/anli", trust_remote_code=True)

    samples = []
    for split in ["train_r1", "train_r2", "train_r3"]:
        if split not in ds:
            continue
        for row in ds[split]:
            label = nli_to_attribution(row["label"])
            samples.append({
                "claim": row["hypothesis"],
                "source": row["premise"],
                "label": label,
                "split": split,
            })

    # Balance and subsample
    attr = [s for s in samples if s["label"] == "Attributable"]
    not_attr = [s for s in samples if s["label"] == "Not Attributable"]
    print(f"  ANLI raw: {len(attr)} Attributable, {len(not_attr)} Not Attributable")

    # Subsample to max_samples, balanced
    n_per_class = min(max_samples // 2, len(attr), len(not_attr))
    random.shuffle(attr)
    random.shuffle(not_attr)
    samples = attr[:n_per_class] + not_attr[:n_per_class]
    random.shuffle(samples)
    print(f"  ANLI final: {len(samples)} samples ({n_per_class} per class)")
    return samples


def load_clearfacts() -> list:
    """Load ClearFacts evaluation set.

    ClearFacts schema: topic, statement, reference_documents, label (S/N/C), category
    """
    print("Loading ClearFacts...")
    ds = load_dataset("just1nseo/ClearFacts")

    samples = []
    for row in ds["train"]:
        # ClearFacts uses S/N/C labels
        label = nli_to_attribution(row.get("label", ""))
        # reference_documents is a list of strings
        ref_docs = row.get("reference_documents", [])
        source = "\n\n".join(ref_docs) if isinstance(ref_docs, list) else str(ref_docs)
        claim = row.get("statement", "")

        if not claim or not source:
            continue

        samples.append(make_eval_sample(
            claim=claim,
            source=source,
            label=label,
            benchmark="clearfacts",
            subset=row.get("category", row.get("topic", "")),
            metadata={"topic": row.get("topic", ""),
                      "category": row.get("category", "")}
        ))
    print(f"  ClearFacts: {len(samples)} samples")
    return samples


def load_grayfacts() -> list:
    """Load GrayFacts (ambiguous) evaluation set.

    GrayFacts schema: same as ClearFacts but label=AMBIG,
    additional_info has original_label
    """
    print("Loading GrayFacts...")
    ds = load_dataset("just1nseo/GrayFacts")

    samples = []
    for row in ds["train"]:
        # GrayFacts label is "AMBIG" — use original_label for evaluation
        additional = row.get("additional_info", {})
        if isinstance(additional, str):
            try:
                additional = json.loads(additional)
            except:
                additional = {}
        original_label = additional.get("original_label", "N")
        label = nli_to_attribution(original_label)

        ref_docs = row.get("reference_documents", [])
        source = "\n\n".join(ref_docs) if isinstance(ref_docs, list) else str(ref_docs)
        claim = row.get("statement", "")

        if not claim or not source:
            continue

        samples.append(make_eval_sample(
            claim=claim,
            source=source,
            label=label,
            benchmark="grayfacts",
            subset=row.get("category", row.get("topic", "")),
            metadata={"topic": row.get("topic", ""),
                      "category": row.get("category", ""),
                      "ambiguity": True,
                      "original_label": original_label}
        ))
    print(f"  GrayFacts: {len(samples)} samples")
    return samples


def load_aggrefact() -> list:
    """Load LLM-AggreFact (11 sub-benchmarks)."""
    print("Loading LLM-AggreFact...")
    ds = load_dataset("lytang/LLM-AggreFact", trust_remote_code=True)

    samples = []
    for split in ds:
        for row in ds[split]:
            label = nli_to_attribution(row.get("label", ""))
            subset = row.get("dataset", row.get("source", "unknown"))
            samples.append(make_eval_sample(
                claim=row.get("claim", row.get("hypothesis", row.get("output", ""))),
                source=row.get("source_text", row.get("premise", row.get("input", row.get("document", "")))),
                label=label,
                benchmark="aggrefact",
                subset=subset,
            ))
    print(f"  LLM-AggreFact: {len(samples)} samples")

    # Print per-subset distribution
    from collections import Counter
    subset_counts = Counter(s["subset"] for s in samples)
    for sub, cnt in sorted(subset_counts.items()):
        print(f"    {sub}: {cnt}")
    return samples


def load_scifact() -> list:
    """Load SciFact."""
    print("Loading SciFact...")
    ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)

    # SciFact has a claims split and a corpus split
    # Try loading claims
    try:
        claims_ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
        samples = []
        for row in claims_ds.get("validation", claims_ds.get("test", [])):
            label_map = {"SUPPORT": "Attributable", "CONTRADICT": "Not Attributable"}
            for ev in row.get("cited_docs", []):
                gold = label_map.get(row.get("label", ""), "Not Attributable")
                samples.append(make_eval_sample(
                    claim=row.get("claim", ""),
                    source=str(ev),
                    label=gold,
                    benchmark="scifact",
                ))
        print(f"  SciFact: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  SciFact loading error: {e}")
        return []


def load_coverbench() -> list:
    """Load CoverBench."""
    print("Loading CoverBench...")
    try:
        ds = load_dataset("google/coverbench", trust_remote_code=True)
        samples = []
        for row in ds["eval"]:
            label = nli_to_attribution(row.get("label", row.get("answer", "")))
            samples.append(make_eval_sample(
                claim=row.get("claim", row.get("question", row.get("hypothesis", ""))),
                source=row.get("context", row.get("premise", row.get("input", ""))),
                label=label,
                benchmark="coverbench",
                subset=row.get("dataset", ""),
            ))
        print(f"  CoverBench: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  CoverBench loading error: {e}")
        return []


# ============================================================
# Main pipeline
# ============================================================
def prepare_sft_data(anli_samples: list, output_dir: Path):
    """Create SFT training data in chat format."""
    print("\nPreparing SFT data...")

    # Convert to SFT format
    sft_samples = []
    for s in anli_samples:
        sft_samples.append(make_sft_sample(
            claim=s["claim"],
            source=s["source"],
            label=s["label"],
        ))

    # Split train/val
    random.shuffle(sft_samples)
    val_size = min(500, len(sft_samples) // 10)
    train_samples = sft_samples[val_size:]
    val_samples = sft_samples[:val_size]

    # Save
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  SFT train: {len(train_samples)} → {train_path}")
    print(f"  SFT val:   {len(val_samples)} → {val_path}")
    return train_samples, val_samples


def prepare_grpo_data(anli_samples: list, output_dir: Path):
    """Create GRPO training data in veRL parquet format."""
    print("\nPreparing GRPO data...")

    # Use a subset for GRPO (boundary-optimal training benefits from moderate size)
    grpo_samples = []
    for s in anli_samples[:5000]:  # ~5K for GRPO
        grpo_samples.append(make_grpo_sample(
            claim=s["claim"],
            source=s["source"],
            label=s["label"],
        ))

    # Convert to DataFrame and save as parquet
    # veRL expects: data_source, prompt, ability, reward_model, extra_info
    records = []
    for s in grpo_samples:
        records.append({
            "data_source": s["data_source"],
            "prompt": s["prompt"],
            "ability": s["ability"],
            "reward_model": json.dumps(s["reward_model"]),
            "extra_info": json.dumps(s["extra_info"]),
        })

    df = pd.DataFrame(records)

    # Split train/val
    val_size = min(500, len(df) // 10)
    train_df = df.iloc[val_size:]
    val_df = df.iloc[:val_size]

    train_path = output_dir / "grpo_train.parquet"
    val_path = output_dir / "grpo_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"  GRPO train: {len(train_df)} → {train_path}")
    print(f"  GRPO val:   {len(val_df)} → {val_path}")
    return train_df, val_df


def prepare_eval_data(output_dir: Path, skip_large: bool = False):
    """Download and prepare all evaluation datasets."""
    print("\nPreparing evaluation data...")

    datasets = {}

    # ClearFacts (primary evaluation)
    try:
        cf = load_clearfacts()
        save_eval(cf, output_dir / "clearfacts.jsonl")
        datasets["clearfacts"] = cf
    except Exception as e:
        print(f"  ERROR loading ClearFacts: {e}")

    # GrayFacts (ambiguity evaluation)
    try:
        gf = load_grayfacts()
        save_eval(gf, output_dir / "grayfacts.jsonl")
        datasets["grayfacts"] = gf
    except Exception as e:
        print(f"  ERROR loading GrayFacts: {e}")

    # LLM-AggreFact (cross-domain)
    if not skip_large:
        try:
            af = load_aggrefact()
            save_eval(af, output_dir / "aggrefact_all.jsonl")
            # Also save per-subset
            from collections import defaultdict
            by_subset = defaultdict(list)
            for s in af:
                by_subset[s["subset"]].append(s)
            for subset, samples in by_subset.items():
                safe_name = subset.replace("/", "_").replace(" ", "_").lower()
                save_eval(samples, output_dir / f"aggrefact_{safe_name}.jsonl")
            datasets["aggrefact"] = af
        except Exception as e:
            print(f"  ERROR loading LLM-AggreFact: {e}")

    # CoverBench (long-context)
    try:
        cb = load_coverbench()
        save_eval(cb, output_dir / "coverbench.jsonl")
        datasets["coverbench"] = cb
    except Exception as e:
        print(f"  ERROR loading CoverBench: {e}")

    return datasets


def save_eval(samples: list, path: Path):
    """Save evaluation samples to JSONL."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} → {path}")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-only", action="store_true",
                        help="Only prepare SFT training data")
    parser.add_argument("--max-anli", type=int, default=60000,
                        help="Max ANLI samples for SFT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HF_AVAILABLE:
        print("ERROR: `datasets` library required. pip install datasets")
        return

    # Step 1: Load ANLI for training
    anli_samples = load_anli(max_samples=args.max_anli)

    # Step 2: Create SFT data
    prepare_sft_data(anli_samples, OUTPUT_DIR)

    # Step 3: Create GRPO data
    prepare_grpo_data(anli_samples, OUTPUT_DIR)

    if not args.sft_only:
        # Step 4: Download and prepare evaluation datasets
        prepare_eval_data(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"All files saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # Print summary
    print("\nFile listing:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_mb = f.stat().st_size / 1024 / 1024
        lines = sum(1 for _ in open(f)) if f.suffix == ".jsonl" else "parquet"
        print(f"  {f.name:<40} {size_mb:>6.1f}MB  {lines} lines")


if __name__ == "__main__":
    main()
