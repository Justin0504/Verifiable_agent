"""Fair comparison v9: Verifier Agent (self-evolution) vs 6 Baselines.

Key changes from v8:
  - Phase 0 calibration: 20% of benchmark as validation set (real data)
  - Soft R-Zero: sort by uncertainty, keep top-60% hardest (not hard filter)
  - Validate prompt against REAL benchmark data (not synthetic)
  - Verification-skill focused generation (paraphrasing, subtle contradiction)
  - Progressive difficulty: each epoch targets current weaknesses

Paper insights:
  - R-Zero (2508.05004): uncertainty-based data selection, boundary cases
  - Self-Challenging (2506.01716): executable verification of gold labels
  - ReasoningBank (2509.25140): success+failure distillation, top-1 retrieval
  - A-MEM (2502.12110): structured memory with evolution

Usage:
    python scripts/run_fair_comparison.py --benchmark truthfulqa --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from src.baselines.base import BaseBaseline, BaselineResult
from src.baselines.cove import CoVeBaseline
from src.baselines.direct_prompting import DirectPromptingBaseline
from src.baselines.factscore_baseline import FActScoreBaseline
from src.baselines.retrieve_nli import RetrieveNLIBaseline
from src.baselines.safe_baseline import SAFEBaseline
from src.baselines.selfcheck import SelfCheckGPTBaseline
from src.benchmarks import (
    FEVERLoader, HaluEvalLoader, MuSiQueLoader, SciFactLoader, TruthfulQALoader,
)
from src.benchmarks.base import BenchmarkSample
from src.data.schema import AtomicClaim, ClaimLabel
from src.evolution.reasoning_bank import ReasoningBank
from src.llm.openai_llm import OpenAILLM

BENCHMARK_LOADERS = {
    "fever": FEVERLoader, "truthfulqa": TruthfulQALoader,
    "scifact": SciFactLoader, "halueval": HaluEvalLoader, "musique": MuSiQueLoader,
}
BASELINE_CLASSES = {
    "direct_prompting": DirectPromptingBaseline, "factscore": FActScoreBaseline,
    "safe": SAFEBaseline, "cove": CoVeBaseline,
    "selfcheck_gpt": SelfCheckGPTBaseline, "retrieve_nli": RetrieveNLIBaseline,
}


# ================================================================
# Synthetic Data Generation
# ================================================================

SYNTH_VERIFICATION_PROMPT = """\
Generate {n} claim-evidence pairs for training a fact verification system.
Each pair tests a specific VERIFICATION SKILL (not just knowledge).

Categories to cover:
1. PARAPHRASING (label: S) — claim restates evidence in different words
2. SUBTLE CONTRADICTION (label: C) — claim changes one key detail (number, name, date)
3. TOPIC OVERLAP (label: N) — claim and evidence are about same topic but different facts
4. PARTIAL MATCH (label: S) — claim is supported but uses less precise language
5. NEGATION FLIP (label: C) — claim negates what evidence says
6. IRRELEVANT (label: N) — claim is about a completely different subject
{weakness_guidance}

For each, provide:
- "claim": A factual statement (1-2 sentences)
- "evidence": A reference passage (2-3 sentences)
- "gold_label": "S", "C", or "N"
- "skill": which verification skill this tests

Output JSON array of {n} examples:
[{{"claim": "...", "evidence": "...", "gold_label": "S|C|N", "skill": "..."}}]
"""

HARD_ADVERSARIAL_PROMPT = """\
Generate {n} ADVERSARIAL verification cases that exploit these specific weaknesses:
{patterns}

Make each case as tricky as possible:
- For S→N errors: make evidence support the claim but use very different wording
- For N→C errors: make claim seem related to evidence but actually about different entity/fact
- For S→C errors: make claim very similar to evidence except it's actually consistent

Output JSON array:
[{{"claim": "...", "evidence": "...", "gold_label": "S|C|N", "skill": "adversarial"}}]
"""

VERIFY_GOLD_PROMPT = """\
Is the following gold label correct?

Claim: {claim}
Evidence: {evidence}
Assigned label: {label}

A label is correct if:
- S: the evidence SUPPORTS or CONFIRMS the claim (paraphrase, entailment, consistent facts)
- C: the evidence CONTRADICTS the claim (conflicting facts, opposite assertions)
- N: the evidence does NOT ADDRESS the claim (different topic, insufficient info)

Reply with JSON: {{"correct": true/false, "reasoning": "brief"}}
"""


def generate_verification_data(llm, n=20, seed=42, weakness_guidance=""):
    """Generate data focused on verification SKILLS, not just knowledge."""
    result = llm.generate(SYNTH_VERIFICATION_PROMPT.format(
        n=n, weakness_guidance=weakness_guidance))
    data = []
    for i, item in enumerate(_parse_json_array(result.text)):
        label = item.get("gold_label", "").upper().strip()
        if item.get("claim") and item.get("evidence") and label in ("S", "C", "N"):
            data.append({
                "id": f"synth_{seed}_{i}", "claim": item["claim"],
                "evidence": item["evidence"], "gold_label": label,
                "skill": item.get("skill", "unknown"), "source": "synth",
            })
    return data


def generate_hard_adversarial(llm, n=10, error_patterns=None, seed=42):
    """Generate adversarial data targeting specific weakness patterns."""
    if not error_patterns:
        return []
    result = llm.generate(HARD_ADVERSARIAL_PROMPT.format(
        n=n, patterns="\n".join(f"- {p}" for p in error_patterns[:5])))
    data = []
    for i, item in enumerate(_parse_json_array(result.text)):
        label = item.get("gold_label", "").upper().strip()
        if item.get("claim") and item.get("evidence") and label in ("S", "C", "N"):
            data.append({
                "id": f"adv_{seed}_{i}", "claim": item["claim"],
                "evidence": item["evidence"], "gold_label": label,
                "skill": "adversarial", "source": "adversarial",
            })
    return data


def verify_gold_labels(llm, data: list[dict]) -> list[dict]:
    """Self-Challenging: verify each example's gold label, discard bad ones."""
    verified = []
    for item in data:
        prompt = VERIFY_GOLD_PROMPT.format(
            claim=item["claim"][:300], evidence=item["evidence"][:300],
            label=item["gold_label"])
        try:
            result = llm.generate(prompt)
            text = result.text.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end > 0:
                verdict = json.loads(text[start:end])
                if verdict.get("correct", False):
                    verified.append(item)
                    continue
        except Exception:
            pass
    return verified


# ================================================================
# Soft R-Zero: Sort by uncertainty, keep hardest
# ================================================================

def soft_rzero_filter(matcher, data: list[dict], n_samples=3, keep_ratio=0.6) -> list[dict]:
    """Soft R-Zero: instead of hard filter (which removes everything),
    sort by uncertainty and keep the top keep_ratio fraction.

    Uncertainty = how close p̂ is to 0.5 (most uncertain = most informative).
    """
    scored = []
    for item in data:
        votes = []
        for _ in range(n_samples):
            claim = AtomicClaim(id=item["id"], text=item["claim"])
            matched = matcher.match(claim, item["evidence"])
            pred = matched.label.value if matched.label else "N"
            votes.append(pred)

        p_hat = sum(1 for v in votes if v == item["gold_label"]) / len(votes)
        uncertainty = 1.0 - abs(p_hat - 0.5) * 2  # 1.0 = max uncertain, 0.0 = fully certain
        item["uncertainty"] = uncertainty
        item["p_hat"] = p_hat
        item["vote_distribution"] = dict(Counter(votes))
        scored.append(item)

    # Sort by uncertainty (highest first = most informative)
    scored.sort(key=lambda x: -x["uncertainty"])

    # Keep top fraction, but at least 5 items
    n_keep = max(5, int(len(scored) * keep_ratio))
    return scored[:n_keep]


# ================================================================
# Claim-level Majority Vote Aggregation
# ================================================================

def aggregate_claim_labels(claim_labels: list[str], claim_confidences: list[float]) -> tuple[str, float]:
    """Confidence-weighted majority vote across claims.

    Key rules:
    1. Single high-confidence C (>=0.8) → C (catches contradictions)
    2. Otherwise: confidence-weighted majority vote
    3. Tie-breaking: C > N > S (conservative for hallucination detection)
    """
    if not claim_labels:
        return "N", 0.5

    # Rule 1: High-confidence C dominates
    if "C" in claim_labels:
        c_confs = [conf for l, conf in zip(claim_labels, claim_confidences) if l == "C"]
        if max(c_confs) >= 0.8:
            return "C", max(c_confs)

    # Rule 2: Confidence-weighted vote
    label_weight = {}
    for label, conf in zip(claim_labels, claim_confidences):
        label_weight[label] = label_weight.get(label, 0) + conf

    # Tie-breaking priority
    priority = {"C": 2, "N": 1, "S": 0}
    predicted = max(label_weight, key=lambda l: (label_weight[l], priority.get(l, 0)))
    avg_conf = sum(claim_confidences) / len(claim_confidences)
    return predicted, avg_conf


# ================================================================
# Verifier Prediction
# ================================================================

def predict_synthetic(matcher, item, rb=None, epoch=0):
    claim = AtomicClaim(id=item["id"], text=item["claim"])
    if rb and epoch > 0:
        relevant = rb.retrieve_relevant(claim.text, top_k=1)
        if relevant:
            claim.metadata = {"reasoning_rules": rb.format_for_prompt(relevant),
                              "rule_ids": [r.id for r in relevant]}
    matched = matcher.match(claim, item["evidence"])
    pred = matched.label.value if matched.label else "N"
    if rb and matched.metadata:
        for rid in matched.metadata.get("rule_ids", []):
            rb.record_usage(rid, pred == item["gold_label"])
    return {"id": item["id"], "claim": item["claim"], "evidence": item["evidence"][:200],
            "gold_label": item["gold_label"], "predicted_label": pred,
            "confidence": matched.confidence, "correct": pred == item["gold_label"]}


def predict_benchmark(matcher, sample, rb=None, epoch=0):
    evidence_text = "\n\n".join(sample.evidence) if sample.evidence else ""
    claims_text = sample.claims if sample.claims else [sample.question]
    claims = [AtomicClaim(id=f"{sample.id}_c{i}", text=c) for i, c in enumerate(claims_text)]

    if rb and epoch > 0:
        for claim in claims:
            relevant = rb.retrieve_relevant(claim.text, top_k=1)
            if relevant:
                claim.metadata = {"reasoning_rules": rb.format_for_prompt(relevant),
                                  "rule_ids": [r.id for r in relevant]}

    verified = [matcher.match(c, evidence_text) for c in claims]
    claim_labels = [c.label.value if c.label else "N" for c in verified]
    claim_confs = [c.confidence for c in verified]

    predicted, avg_conf = aggregate_claim_labels(claim_labels, claim_confs)

    if rb:
        for c_obj, pred_l in zip(verified, claim_labels):
            if c_obj.metadata:
                for rid in c_obj.metadata.get("rule_ids", []):
                    rb.record_usage(rid, pred_l == sample.gold_label)

    return BaselineResult(
        sample_id=sample.id, predicted_label=predicted, gold_label=sample.gold_label,
        confidence=avg_conf,
        claims=[{"claim": c.text, "label": l} for c, l in zip(verified, claim_labels)],
        claim_labels=claim_labels,
        claim_gold_labels=[sample.gold_label] * len(claims_text),
        metadata={"method": "verifier_agent"},
    )


def predict_majority(matcher, sample, rb=None, epoch=0, n_votes=3):
    """R-Zero: majority vote at test time for robustness."""
    votes, confs = [], []
    for _ in range(n_votes):
        r = predict_benchmark(matcher, sample, rb, epoch)
        votes.append(r.predicted_label)
        confs.append(r.confidence)
    label_conf = {}
    for v, c in zip(votes, confs):
        label_conf[v] = label_conf.get(v, 0) + c
    majority = max(label_conf, key=label_conf.get)
    return BaselineResult(
        sample_id=sample.id, predicted_label=majority, gold_label=sample.gold_label,
        confidence=sum(confs) / len(confs),
        metadata={"method": "verifier_agent_ensemble", "votes": dict(Counter(votes))},
    )


# ================================================================
# Error Analysis + Prompt Refinement
# ================================================================

def analyze_errors(preds):
    total = len(preds)
    correct = sum(1 for p in preds if p["correct"])
    acc = correct / total if total else 0
    labels = ["S", "C", "N"]
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for p in preds:
        g, pr = p["gold_label"], p["predicted_label"]
        if g in confusion and pr in confusion[g]:
            confusion[g][pr] += 1
    errors = []
    for g in labels:
        for p in labels:
            if g != p and confusion[g][p] > 0:
                errors.append({"from": g, "to": p, "count": confusion[g][p]})
    errors.sort(key=lambda x: -x["count"])
    return {"accuracy": acc, "total": total, "correct": correct,
            "confusion_matrix": confusion, "top_errors": errors[:5],
            "error_examples": [p for p in preds if not p["correct"]][:8]}


REFINE_PROMPT = """\
You are optimizing a claim verification system prompt. Task: given a claim and evidence, label as:
- S (Supported): evidence confirms or entails the claim
- C (Contradicted): evidence conflicts with the claim
- N (Not enough info): evidence doesn't address the claim

CURRENT PROMPT:
---
{current}
---

PERFORMANCE ON REAL BENCHMARK ({total} samples, {acc:.0%} accuracy):
{errors}

CONCRETE ERROR EXAMPLES:
{examples}

Write an IMPROVED system prompt that:
1. Stays under 250 words
2. Addresses the specific error patterns above
3. Has clear decision rules for S vs C vs N boundaries
4. Handles multi-claim evidence (some claims may be supported while others are not)
5. Uses structured output: {{"label": "S"/"C"/"N", "confidence": <float>, "reasoning": "<brief>", "evidence_snippet": "<key quote>"}}

IMPORTANT: The most common error is confusing N with C or S. Be explicit about when to use N.

Write ONLY the improved prompt:"""


def refine_prompt(llm, current, analysis):
    err_lines = [f"- {e['from']}→{e['to']} ({e['count']}x)" for e in analysis.get("top_errors", [])]
    if not err_lines:
        return current
    ex_lines = []
    for p in analysis.get("error_examples", [])[:5]:
        ex_lines.append(
            f'- Claim: "{p["claim"][:80]}"\n'
            f'  Evidence: "{p["evidence"][:80]}"\n'
            f"  Predicted: {p['predicted_label']}, Gold: {p['gold_label']}"
        )
    result = llm.generate(REFINE_PROMPT.format(
        current=current or "(none — epoch 0 baseline, no system prompt)",
        total=analysis["total"], acc=analysis["accuracy"],
        errors="\n".join(err_lines), examples="\n".join(ex_lines)))
    new = result.text.strip()
    if '"S"' in new and '"C"' in new and '"N"' in new and "label" in new:
        return new
    return current


# ================================================================
# ReasoningBank: success + failure distillation
# ================================================================

def distill_rules(rb, preds, epoch):
    if not rb:
        return 0
    trajectories = []
    # More failure trajectories for richer rule distillation
    for p in preds:
        if not p["correct"] and len(trajectories) < 5:
            trajectories.append({
                "claim": p["claim"], "evidence": p["evidence"],
                "predicted_label": p["predicted_label"], "gold_label": p["gold_label"],
                "confidence": p["confidence"],
                "context": f"FAILURE: predicted {p['predicted_label']} but gold is {p['gold_label']}",
            })
    for p in preds:
        if p["correct"] and p["confidence"] >= 0.8 and len(trajectories) < 8:
            trajectories.append({
                "claim": p["claim"], "evidence": p["evidence"],
                "predicted_label": p["predicted_label"], "gold_label": p["gold_label"],
                "confidence": p["confidence"],
                "context": f"SUCCESS: correctly predicted {p['predicted_label']} with high confidence",
            })
    if trajectories:
        rules = rb.distill_batch(trajectories, epoch=epoch)
        rb._evict()
        rb.save()
        return len(rules)
    return 0


# ================================================================
# Helpers
# ================================================================

def _parse_json_array(text):
    text = text.strip()
    s, e = text.find("["), text.rfind("]") + 1
    if s == -1 or e == 0: return []
    try: return json.loads(text[s:e])
    except: return []


def load_samples(name, limit, split):
    loader = BENCHMARK_LOADERS[name]()
    try:
        samples = loader.load(split=split, limit=limit)
    except Exception as e:
        print(f"  HuggingFace failed ({e}), manual fallback...")
        samples = loader.load_manual_sample(limit=limit or 50)
    return [s for s in samples if s.gold_label in ("S", "C", "N")]


def eval_on_samples(matcher, samples, rb=None, epoch=0):
    """Evaluate verifier accuracy on benchmark samples. Returns (accuracy, results_list)."""
    results = []
    for s in samples:
        r = predict_benchmark(matcher, s, rb, epoch)
        results.append({
            "id": s.id, "predicted_label": r.predicted_label, "gold_label": s.gold_label,
            "correct": r.predicted_label == s.gold_label, "confidence": r.confidence,
            "claim": s.claims[0][:100] if s.claims else s.question[:100],
            "evidence": s.evidence[0][:100] if s.evidence else "",
        })
    acc = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    return acc, results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="truthfulqa")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--synth-per-epoch", type=int, default=40)
    parser.add_argument("--hard-per-epoch", type=int, default=20)
    parser.add_argument("--majority-votes", type=int, default=3)
    parser.add_argument("--rzero-samples", type=int, default=3)
    parser.add_argument("--rzero-keep", type=float, default=0.8)
    parser.add_argument("--cal-ratio", type=float, default=0.2)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--baselines", default="direct_prompting,factscore,safe,cove,selfcheck_gpt,retrieve_nli")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"=== Fair Comparison v9: {args.benchmark} ===")
    print(f"Epochs: {args.epochs} | Votes: {args.majority_votes}x | Cal: {args.cal_ratio:.0%}")
    print(f"Key: real-data validation, soft R-Zero, verification-skill training")
    print()

    # Load ALL samples, then split into calibration + test
    all_samples = load_samples(args.benchmark, args.test_limit, args.split)
    random.Random(42).shuffle(all_samples)

    n_cal = max(10, int(len(all_samples) * args.cal_ratio))
    cal_samples = all_samples[:n_cal]
    test_samples = all_samples[n_cal:]
    cal_dist = Counter(s.gold_label for s in cal_samples)
    test_dist = Counter(s.gold_label for s in test_samples)
    print(f"Calibration: {len(cal_samples)} samples {dict(cal_dist)}")
    print(f"Test: {len(test_samples)} samples {dict(test_dist)}")
    print()

    llm = OpenAILLM(model=args.model, temperature=0.2)
    output_dir = args.output or f"results/fair_comparison_{args.benchmark}_v9"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rb_path = Path(f"{output_dir}/reasoning_bank.json")
    if rb_path.exists(): rb_path.unlink()
    rb = ReasoningBank(path=str(rb_path), llm=llm)

    from src.verifier.evidence_matcher import EvidenceMatcher
    matcher = EvidenceMatcher(llm)

    # ============================================================
    # Phase 0: Baseline calibration (epoch 0, no evolution)
    # ============================================================
    print("=" * 60)
    print("PHASE 0: Baseline Calibration")
    print("=" * 60)

    cal_acc_0, cal_results_0 = eval_on_samples(matcher, cal_samples, rb, epoch=0)
    cal_errors_0 = [r for r in cal_results_0 if not r["correct"]]
    print(f"  Epoch 0 calibration: {cal_acc_0:.1%} ({sum(1 for r in cal_results_0 if r['correct'])}/{len(cal_results_0)})")

    # Analyze calibration errors → weakness patterns
    error_type_counts = Counter()
    for r in cal_errors_0:
        error_type_counts[f"{r['gold_label']}→{r['predicted_label']}"] += 1
    weakness_patterns = []
    for pattern, count in error_type_counts.most_common(5):
        print(f"    Error: {pattern} ({count}x)")
        fr, to = pattern.split("→")
        if fr == "S" and to == "N":
            weakness_patterns.append("Evidence supports claim but uses different wording (S misclassified as N)")
        elif fr == "S" and to == "C":
            weakness_patterns.append("Evidence supports claim but model sees irrelevant differences as contradictions (S misclassified as C)")
        elif fr == "C" and to == "S":
            weakness_patterns.append("Evidence contradicts claim but model misses the contradiction (C misclassified as S)")
        elif fr == "C" and to == "N":
            weakness_patterns.append("Evidence contradicts claim but model says not enough info (C misclassified as N)")
        elif fr == "N" and to == "S":
            weakness_patterns.append("Evidence doesn't address claim but model sees topic overlap as support (N misclassified as S)")
        elif fr == "N" and to == "C":
            weakness_patterns.append("Evidence doesn't address claim but model sees topic overlap as contradiction (N misclassified as C)")
        else:
            weakness_patterns.append(f"{pattern} confusion")

    # ============================================================
    # Phase 1: Self-Evolution
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Self-Evolution")
    print("=" * 60)

    curve = []
    best_cal_acc = cal_acc_0
    best_prompt = ""

    for epoch in range(args.epochs):
        print(f"\n  --- Epoch {epoch+1}/{args.epochs} ---")
        matcher.epoch = epoch
        t0 = time.time()

        # 1. Generate verification-skill data
        weakness_guidance = ""
        if weakness_patterns:
            weakness_guidance = (
                "\nFocus on generating examples that test these specific weaknesses:\n"
                + "\n".join(f"- {p}" for p in weakness_patterns[:3])
            )
        synth = generate_verification_data(llm, n=args.synth_per_epoch,
                                            seed=epoch*100, weakness_guidance=weakness_guidance)
        print(f"    Synthetic: {len(synth)}")

        # 1b. Hard adversarial targeting current weaknesses
        if weakness_patterns:
            hard = generate_hard_adversarial(llm, n=args.hard_per_epoch,
                                              error_patterns=weakness_patterns, seed=epoch*100+50)
            synth.extend(hard)
            if hard:
                print(f"    + {len(hard)} adversarial")

        # 2. Self-Challenging: verify gold labels
        before = len(synth)
        synth = verify_gold_labels(llm, synth)
        retained = len(synth) / before if before else 0
        print(f"    Gold verified: {before} → {len(synth)} ({retained:.0%})")

        if len(synth) < 5:
            print(f"    SKIP: too few ({len(synth)})")
            curve.append({"epoch": epoch+1, "skip": True, "cal_acc": best_cal_acc})
            continue

        # 3. Soft R-Zero: sort by uncertainty, keep hardest
        if epoch >= 1 and args.rzero_samples > 1:
            before_rz = len(synth)
            synth = soft_rzero_filter(matcher, synth, n_samples=args.rzero_samples,
                                       keep_ratio=args.rzero_keep)
            print(f"    Soft R-Zero: {before_rz} → {len(synth)} (kept hardest)")

        if len(synth) < 5:
            print(f"    SKIP: too few after filter")
            curve.append({"epoch": epoch+1, "skip": True, "cal_acc": best_cal_acc})
            continue

        # 4. Train on synthetic data
        train_preds = [predict_synthetic(matcher, item, rb, epoch) for item in synth]
        synth_analysis = analyze_errors(train_preds)
        print(f"    Synth train: {synth_analysis['accuracy']:.1%} ({synth_analysis['correct']}/{synth_analysis['total']})")
        for e in synth_analysis["top_errors"][:2]:
            print(f"      {e['from']}→{e['to']}: {e['count']}x")

        # 5. Prompt refinement with REAL benchmark validation gate
        old_prompt = matcher.evolved_prompt or ""

        # Combine synth errors + calibration errors for prompt refinement
        # Use calibration errors as primary signal (real data)
        combined_analysis = synth_analysis.copy()
        if cal_errors_0:
            combined_analysis["error_examples"] = (
                [r for r in cal_results_0 if not r["correct"]][:4]
                + synth_analysis.get("error_examples", [])[:4]
            )

        new_prompt = refine_prompt(llm, old_prompt, combined_analysis)

        if new_prompt != old_prompt:
            # Validate on REAL calibration data
            matcher.evolved_prompt = new_prompt
            matcher.epoch = epoch + 1
            new_cal_acc, new_cal_results = eval_on_samples(matcher, cal_samples, rb, epoch+1)

            # Accept if improves OR ties (allows lateral exploration)
            # Also accept slight regression (-1%) every 5 epochs to escape local optima
            stale_epochs = epoch - max((i for i, c in enumerate(curve) if not c.get("skip")), default=0)
            tolerance = 0.01 if stale_epochs >= 4 else 0.0
            if new_cal_acc >= best_cal_acc - tolerance:
                if new_cal_acc > best_cal_acc:
                    print(f"    Prompt ACCEPTED (improved): cal {best_cal_acc:.1%} → {new_cal_acc:.1%}")
                elif new_cal_acc == best_cal_acc:
                    print(f"    Prompt ACCEPTED (lateral): cal {best_cal_acc:.1%} → {new_cal_acc:.1%}")
                else:
                    print(f"    Prompt ACCEPTED (exploration): cal {best_cal_acc:.1%} → {new_cal_acc:.1%}")
                best_cal_acc = max(best_cal_acc, new_cal_acc)
                best_prompt = new_prompt

                # Update weakness patterns from new calibration results
                cal_errors_0 = [r for r in new_cal_results if not r["correct"]]
                error_type_counts = Counter()
                for r in cal_errors_0:
                    error_type_counts[f"{r['gold_label']}→{r['predicted_label']}"] += 1
                weakness_patterns = []
                for pattern, count in error_type_counts.most_common(5):
                    fr, to = pattern.split("→")
                    if fr == "S" and to == "N":
                        weakness_patterns.append("Evidence supports claim but uses different wording (S→N)")
                    elif fr == "S" and to == "C":
                        weakness_patterns.append("Evidence supports claim but seen as contradiction (S→C)")
                    elif fr == "C" and to == "S":
                        weakness_patterns.append("Evidence contradicts claim but missed (C→S)")
                    elif fr == "C" and to == "N":
                        weakness_patterns.append("Contradiction labeled as insufficient info (C→N)")
                    elif fr == "N" and to == "S":
                        weakness_patterns.append("Unrelated evidence seen as support (N→S)")
                    elif fr == "N" and to == "C":
                        weakness_patterns.append("Unrelated evidence seen as contradiction (N→C)")
                    else:
                        weakness_patterns.append(f"{pattern} confusion")
            else:
                # Reject prompt, revert
                matcher.evolved_prompt = old_prompt
                matcher.epoch = epoch
                print(f"    Prompt REJECTED: cal {best_cal_acc:.1%} → {new_cal_acc:.1%}")
        else:
            print(f"    No prompt change (synth {synth_analysis['accuracy']:.0%})")

        # 6. ReasoningBank distillation
        n_rules = distill_rules(rb, train_preds, epoch)
        if n_rules:
            print(f"    Rules: +{n_rules}, total={rb.stats()['total_rules']}")

        elapsed = time.time() - t0
        curve.append({
            "epoch": epoch+1, "synth_accuracy": synth_analysis["accuracy"],
            "synth_n": synth_analysis["total"], "cal_accuracy": best_cal_acc,
            "prompt_len": len(matcher.evolved_prompt),
            "rb_rules": rb.stats()["total_rules"],
            "n_weaknesses": len(weakness_patterns),
            "time_s": round(elapsed),
        })
        print(f"    Cal best: {best_cal_acc:.1%} | Time: {elapsed:.0f}s")

    # Use best prompt
    if best_prompt:
        matcher.evolved_prompt = best_prompt
        matcher.epoch = args.epochs
        print(f"\n  Best prompt: cal={best_cal_acc:.1%}, {len(best_prompt)} chars")
    else:
        print(f"\n  No prompt evolution (baseline: {cal_acc_0:.1%})")
    print(f"  ReasoningBank: {rb.stats()}")

    # Final calibration accuracy
    final_cal_acc, _ = eval_on_samples(matcher, cal_samples, rb, args.epochs)
    print(f"  Final calibration accuracy: {final_cal_acc:.1%}")

    # ============================================================
    # Phase 2: Test
    # ============================================================
    print("\n" + "=" * 60)
    n_votes = args.majority_votes
    print(f"PHASE 2: Test ({n_votes}x vote, majority-agg)")
    print("=" * 60)

    t0 = time.time()
    test_results = []
    for i, s in enumerate(test_samples):
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(test_samples)}]...")
        r = predict_majority(matcher, s, rb, args.epochs, n_votes) if n_votes > 1 \
            else predict_benchmark(matcher, s, rb, args.epochs)
        test_results.append(r)

    vm = BaseBaseline.compute_metrics(test_results)
    print(f"\n  Verifier Agent: {vm['accuracy']:.1%} (F1={vm['macro_f1']:.3f})")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ============================================================
    # Phase 3: Baselines
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Baselines (same test set)")
    print("=" * 60)

    all_metrics = {"verifier_agent": vm}
    for bl_name in [b.strip() for b in args.baselines.split(",")]:
        print(f"\n  {bl_name}...")
        bl_cls = BASELINE_CLASSES[bl_name]
        bl = bl_cls(llm, n_samples=3) if bl_name == "selfcheck_gpt" else bl_cls(llm)
        t0 = time.time()
        bl_results = [bl.verify_sample(s) for s in test_samples]
        bm = BaseBaseline.compute_metrics(bl_results)
        elapsed = time.time() - t0
        print(f"    {bm['accuracy']:.1%} (F1={bm['macro_f1']:.3f}, {elapsed:.0f}s)")
        all_metrics[bl_name] = bm

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {args.benchmark} | {len(test_samples)} test + {len(cal_samples)} cal | v9")
    print("=" * 70)

    ranked = sorted(all_metrics.items(), key=lambda x: -x[1]["accuracy"])
    print(f"\n{'Method':<25} {'Accuracy':>10} {'F1':>8} {'Rank':>6}")
    print("-" * 53)
    for rank, (name, m) in enumerate(ranked, 1):
        marker = " ***" if name == "verifier_agent" else ""
        print(f"{name:<25} {m['accuracy']:>9.1%} {m['macro_f1']:>8.3f} {rank:>5}{marker}")

    v_rank = next(i for i, (n, _) in enumerate(ranked, 1) if n == "verifier_agent")
    if v_rank == 1:
        print(f"\n  >>> VERIFIER AGENT WINS (self-evolution effective) <<<")
    else:
        gap = ranked[0][1]["accuracy"] - vm["accuracy"]
        print(f"\n  Rank #{v_rank} (behind {ranked[0][0]} by {gap:.1%})")

    # Evolution summary
    print(f"\n  Evolution: {cal_acc_0:.1%} (epoch 0) → {best_cal_acc:.1%} (best cal)")
    print(f"  Improvement: +{best_cal_acc - cal_acc_0:.1%}")

    # Save
    with open(Path(output_dir) / "comparison.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(), "version": "v9",
            "config": vars(args),
            "evolution_curve": curve,
            "baseline_cal_accuracy": cal_acc_0,
            "best_cal_accuracy": best_cal_acc,
            "best_prompt": best_prompt[:500] if best_prompt else "",
            "metrics": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                            "confusion_matrix": v.get("confusion_matrix", {})}
                        for k, v in all_metrics.items()},
            "ranking": [(n, m["accuracy"]) for n, m in ranked],
            "verifier_rank": v_rank,
        }, f, indent=2)
    with open(Path(output_dir) / "evolution_curve.json", "w") as f:
        json.dump(curve, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
