"""Generate weakness-targeted adversarial samples for self-evolution.

Reads the weakness profile from analyze_failures.py and generates
adversarial (claim, source, label) triples targeting the model's
specific weaknesses. Then annotates them with structured output
using the teacher model.

This is the PROBE phase of the self-evolution loop.

Usage:
    python scripts/generate_adversarial_probes.py \
        --weakness-profile results/analysis/weakness_profile.json \
        --output-dir data/attribution/adversarial_round1 \
        --num-samples 2000

    # With annotation for SFT/GRPO
    python scripts/generate_adversarial_probes.py \
        --weakness-profile results/analysis/weakness_profile.json \
        --output-dir data/attribution/adversarial_round1 \
        --num-samples 2000 --annotate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verifier.seva_format import (
    ERROR_TYPES,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_USER_TEMPLATE,
    SEVA_SYSTEM_PROMPT,
    SEVA_USER_TEMPLATE,
)
from src.llm.openai_llm import OpenAILLM

# Reuse validation from SFT data generation
from scripts.generate_seva_sft_data import validate_structured_output, extract_json


# ============================================================
# Adversarial probe generation prompt
# ============================================================

PROBE_SYSTEM_PROMPT = """\
You are an adversarial data generator for fact attribution verification.
Your job is to create challenging (claim, source) pairs that test a \
verifier's ability to detect specific types of attribution errors.

You will be given a target error type and difficulty level.

Generate a JSON object with EXACTLY these fields:
{
  "claim": "A factual claim (1-3 sentences)",
  "source": "A source document (2-5 sentences) that the claim references",
  "label": "Attributable" or "Not Attributable",
  "error_type": "the error type (ONLY if Not Attributable, omit otherwise)",
  "difficulty": "easy/medium/hard",
  "domain": "science/politics/health/finance/technology/sports/history"
}

Rules:
- For "Not Attributable": the claim must be CLOSE to the source but differ \
in the specific way described by the error type.
- For hard difficulty: make the difference VERY subtle, requiring careful reading.
- For "Attributable": the claim must be fully supported. Include NO errors.
- Sources should be realistic (Wikipedia/news/academic style).
- Vary domains. Output ONLY the JSON object."""


def make_probe_prompt(error_type: str, difficulty: str,
                      weakness_context: str = "") -> str:
    """Create prompt for generating one adversarial sample."""
    if error_type == "safe_attribution":
        return (
            "Generate an ATTRIBUTABLE (claim, source) pair. "
            "The claim must be fully and clearly supported by the source. "
            f"Difficulty: {difficulty}. "
            "Make the source detailed enough that attribution is unambiguous."
        )

    if error_type == "subtle_non_attribution":
        return (
            "Generate a NOT ATTRIBUTABLE (claim, source) pair. "
            "The claim should be very close to the source but differ in one "
            "subtle detail. Choose any error type. Difficulty: hard."
        )

    error_desc = ERROR_TYPES.get(error_type, error_type)
    prompt = (
        f"Generate a NOT ATTRIBUTABLE (claim, source) pair.\n"
        f"Error type: {error_type}\n"
        f"Error description: {error_desc}\n"
        f"Difficulty: {difficulty}\n"
    )

    if difficulty == "hard":
        prompt += (
            "Make the difference VERY subtle. The claim should look almost "
            "identical to what the source says, with only a small but meaningful "
            "difference.\n"
        )

    if weakness_context:
        prompt += f"\nVerifier weakness context: {weakness_context}\n"

    return prompt


# ============================================================
# Generation
# ============================================================
def generate_probe(llm: OpenAILLM, error_type: str, difficulty: str,
                   weakness_context: str = "", max_retries: int = 2) -> dict | None:
    """Generate one adversarial (claim, source, label) triple."""
    prompt = make_probe_prompt(error_type, difficulty, weakness_context)

    for attempt in range(max_retries + 1):
        try:
            response = llm.generate(prompt, system=PROBE_SYSTEM_PROMPT)
            text = response.text.strip()

            # Remove markdown fences
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = text.replace("```", "")

            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                continue

            obj = json.loads(text[start:end])

            # Validate basic structure
            if not obj.get("claim") or not obj.get("source"):
                continue
            if obj.get("label") not in ("Attributable", "Not Attributable"):
                continue
            if len(obj["claim"]) < 20 or len(obj["source"]) < 50:
                continue

            # For NA samples, ensure error_type
            if obj["label"] == "Not Attributable" and not obj.get("error_type"):
                obj["error_type"] = error_type

            return obj

        except (json.JSONDecodeError, Exception):
            if attempt == max_retries:
                return None

    return None


def annotate_probe(llm: OpenAILLM, probe: dict,
                   max_retries: int = 2) -> dict | None:
    """Annotate a probe with full structured SEVA output using teacher."""
    claim = probe["claim"]
    source = probe["source"]
    label = probe["label"]

    prompt = TEACHER_USER_TEMPLATE.format(claim=claim, source=source, label=label)

    for attempt in range(max_retries + 1):
        try:
            response = llm.generate(prompt, system=TEACHER_SYSTEM_PROMPT)
            obj = extract_json(response.text)
            if obj is None:
                continue
            valid, _ = validate_structured_output(obj, label)
            if valid:
                return obj
        except Exception:
            pass

    return None


def plan_batch(weakness_profile: dict, num_samples: int,
               replay_ratio: float = 0.2) -> list[tuple[str, str]]:
    """Plan sample types based on weakness profile.

    Returns list of (error_type, difficulty) pairs.
    """
    weights = dict(weakness_profile.get("targeting_weights", {}))
    if not weights:
        all_types = list(ERROR_TYPES.keys()) + ["safe_attribution"]
        weights = {t: 1.0 / len(all_types) for t in all_types}

    n_new = int(num_samples * (1.0 - replay_ratio))

    # Ensure at least 15% safe attribution
    if "safe_attribution" not in weights:
        weights["safe_attribution"] = 0.15
    elif weights["safe_attribution"] < 0.15:
        weights["safe_attribution"] = 0.15

    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Build plan
    plan = []
    weaknesses = weakness_profile.get("error_type_weaknesses", {})

    for etype, w in weights.items():
        n = max(1, int(n_new * w))
        priority = weaknesses.get(etype, {}).get("priority", 0.5)

        for _ in range(n):
            if priority > 0.6:
                diff = random.choice(["hard", "hard", "medium"])
            elif priority > 0.3:
                diff = random.choice(["hard", "medium", "medium"])
            else:
                diff = random.choice(["medium", "medium", "easy"])
            plan.append((etype, diff))

    random.shuffle(plan)
    return plan[:n_new]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weakness-profile", type=str, required=True)
    parser.add_argument("--hard-samples", type=str, default=None,
                        help="hard_samples.json for replay buffer")
    parser.add_argument("--source-data", type=str, default=None,
                        help="Original training data for replaying hard samples")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--replay-ratio", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate probes with structured SEVA output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weakness profile
    print(f"Loading weakness profile: {args.weakness_profile}")
    with open(args.weakness_profile) as f:
        profile = json.load(f)

    print(f"  Overall: acc={profile.get('overall_accuracy')}, "
          f"F1={profile.get('overall_f1')}, bias={profile.get('bias')}")
    print(f"  Targeting weights:")
    for etype, w in sorted(profile.get("targeting_weights", {}).items(),
                           key=lambda x: x[1], reverse=True):
        print(f"    {etype:30s}  {w:.3f}")

    # Plan batch
    plan = plan_batch(profile, args.num_samples, args.replay_ratio)
    print(f"\nPlan: {len(plan)} new samples")
    type_counts = {}
    for etype, _ in plan:
        type_counts[etype] = type_counts.get(etype, 0) + 1
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype:30s}  {count:>4d}")

    # Init LLM
    llm = OpenAILLM(model=args.model, temperature=0.8)

    # Build weakness context
    weakness_ctx = ""
    weaknesses = profile.get("error_type_weaknesses", {})
    if weaknesses:
        worst = list(weaknesses.items())[:3]
        weakness_ctx = "Verifier struggles with: " + ", ".join(
            f"{k} ({v['accuracy']:.0%} acc)" for k, v in worst
        )

    # Generate probes
    print(f"\nGenerating {len(plan)} adversarial probes...")
    probes = []
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (etype, diff) in enumerate(plan):
            f = pool.submit(generate_probe, llm, etype, diff, weakness_ctx)
            futures[f] = (i, etype, diff)

        for future in as_completed(futures):
            i, etype, diff = futures[future]
            result = future.result()
            if result:
                result["_meta"] = {
                    "target_type": etype,
                    "target_diff": diff,
                    "source": "adversarial_probe",
                }
                probes.append(result)
            else:
                failed += 1

            done = len(probes) + failed
            if done % 100 == 0:
                print(f"  [{done}/{len(plan)}] {len(probes)} ok, "
                      f"{failed} fail ({time.time()-t0:.0f}s)")

    print(f"\n  Generated: {len(probes)} ({failed} failed, "
          f"{time.time()-t0:.0f}s)")

    # --- Contrastive pairs: minimal edits on Attributable probes ---
    positives = [p for p in probes if p.get("label") == "Attributable"]
    if positives:
        print(f"\nGenerating contrastive pairs for {len(positives)} positives...")
        contrastive = []
        contrastive_failed = 0

        CONTRASTIVE_SYSTEM = (
            "You are a precise adversarial editor. Given an attributable (claim, source) pair, "
            "create minimal modifications to the claim that make it NOT attributable. "
            "Change as little as possible — ideally one word, number, or phrase."
        )
        CONTRASTIVE_TMPL = (
            "Original claim (fully supported by source):\n{claim}\n\n"
            "Source:\n{source}\n\n"
            "Generate 2 minimally modified claims that are NOT attributable. "
            "Each should change exactly ONE detail. Use different modification types:\n"
            "- numerical_exaggeration: change a number/quantity\n"
            "- negation_flip: add/remove a negation\n"
            "- scope_inflation: 'some' → 'all', specific → general\n"
            "- temporal_shift: remove/alter temporal qualifier\n"
            "- entity_substitution: swap a name/entity\n\n"
            'Output: JSON array of {{"claim": "...", "error_type": "...", "what_changed": "..."}}'
        )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def gen_contrastive(probe):
                prompt = CONTRASTIVE_TMPL.format(
                    claim=probe["claim"], source=probe["source"][:1500]
                )
                try:
                    resp = llm.generate(prompt, system=CONTRASTIVE_SYSTEM)
                    text = resp.text.strip()
                    s, e = text.find("["), text.rfind("]") + 1
                    if s == -1 or e == 0:
                        return []
                    items = json.loads(text[s:e])
                    results = []
                    for item in items:
                        if "claim" not in item:
                            continue
                        results.append({
                            "claim": item["claim"],
                            "source": probe["source"],
                            "label": "Not Attributable",
                            "error_type": item.get("error_type", ""),
                            "difficulty": "hard",
                            "_meta": {
                                "target_type": "contrastive",
                                "source": "contrastive_pair",
                                "original_claim": probe["claim"],
                                "what_changed": item.get("what_changed", ""),
                            },
                        })
                    return results
                except Exception:
                    return []

            futures = {pool.submit(gen_contrastive, p): p for p in positives}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    contrastive.extend(result)
                else:
                    contrastive_failed += 1

        print(f"  Contrastive pairs: {len(contrastive)} generated")
        probes.extend(contrastive)

    # --- Hard negative mining from previous eval results ---
    if args.hard_samples and Path(args.hard_samples).exists():
        print(f"\nLoading hard negatives from {args.hard_samples}...")
        with open(args.hard_samples) as f:
            hard = json.load(f)
        # Filter: high confidence + wrong prediction
        hard_negatives = [s for s in hard
                         if s.get("confidence", 0) > 0.7 and not s.get("correct", True)]
        # Add as replay samples
        for s in hard_negatives[:int(args.num_samples * args.replay_ratio)]:
            probes.append({
                "claim": s.get("claim", ""),
                "source": s.get("source", ""),
                "label": s.get("gold", s.get("gold_label", "")),
                "error_type": s.get("error_type", ""),
                "difficulty": "hard",
                "_meta": {"source": "hard_negative_replay"},
            })
        print(f"  Added {min(len(hard_negatives), int(args.num_samples * args.replay_ratio))} hard negatives")

    # Save probes
    probes_path = output_dir / "adversarial_probes.jsonl"
    with open(probes_path, "w") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved: {probes_path}")

    # Annotate if requested
    annotated = []
    if args.annotate:
        print(f"\nAnnotating {len(probes)} probes with structured output...")
        teacher = OpenAILLM(model=args.model, temperature=0.3)
        ann_failed = 0
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for i, probe in enumerate(probes):
                f = pool.submit(annotate_probe, teacher, probe)
                futures[f] = (i, probe)

            for future in as_completed(futures):
                i, probe = futures[future]
                annotation = future.result()
                if annotation:
                    sample = {
                        "messages": [
                            {"role": "system", "content": SEVA_SYSTEM_PROMPT},
                            {"role": "user", "content": SEVA_USER_TEMPLATE.format(
                                claim=probe["claim"],
                                source=probe["source"][:2000],
                            )},
                            {"role": "assistant", "content": json.dumps(
                                annotation, ensure_ascii=False
                            )},
                        ],
                        "gold_label": probe["label"],
                        "source": "adversarial",
                        "error_type": probe.get("error_type", ""),
                    }
                    annotated.append(sample)
                else:
                    ann_failed += 1

                done = len(annotated) + ann_failed
                if done % 100 == 0:
                    print(f"  [{done}/{len(probes)}] {len(annotated)} ok")

        print(f"  Annotated: {len(annotated)} ({ann_failed} failed, "
              f"{time.time()-t1:.0f}s)")

        # Save SFT format
        sft_path = output_dir / "adversarial_sft.jsonl"
        with open(sft_path, "w") as f:
            for s in annotated:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved SFT: {sft_path}")

        # Save GRPO parquet
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            records = []
            for s in annotated:
                records.append({
                    "data_source": "adversarial",
                    "prompt": json.dumps([
                        {"role": "system", "content": SEVA_SYSTEM_PROMPT},
                        {"role": "user", "content": s["messages"][1]["content"]},
                    ]),
                    "ability": "fact_attribution",
                    "reward_model": json.dumps({
                        "style": "rule",
                        "ground_truth": {
                            "target": s["gold_label"],
                            "claim": s["messages"][1]["content"].split("Source:")[0].replace("Claim:", "").strip() if "Source:" in s["messages"][1]["content"] else "",
                            "source": s["messages"][1]["content"].split("Source:")[1].split("Verify")[0].strip() if "Source:" in s["messages"][1]["content"] else "",
                        },
                    }),
                    "extra_info": json.dumps({
                        "source": "adversarial",
                        "error_type": s.get("error_type", ""),
                        "index": i,
                    }),
                })

            table = pa.table({k: [r[k] for r in records] for k in records[0]})
            pq_path = output_dir / "adversarial_grpo.parquet"
            pq.write_table(table, pq_path)
            print(f"  Saved GRPO: {pq_path}")
        except ImportError:
            print("  WARN: pyarrow unavailable, skipping parquet")

    # Summary
    print(f"\n{'='*60}")
    print(f"PROBE phase complete")
    print(f"  Probes:     {len(probes)}")
    if annotated:
        print(f"  Annotated:  {len(annotated)}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
