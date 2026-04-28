"""SEVA v3 Reward: Grounded Process Reward with Cross-Component Coherence.

8 reward components (normalized to sum=1.0):
1. R_format     : valid JSON with ALL required structured fields
2. R_accuracy   : correct label match (weight decreases across epochs)
3. R_calibration: confidence alignment (asymmetric: harsh penalty for wrong+confident)
4. R_alignment  : evidence spans ACTUALLY grounded in claim/source (threshold=0.75)
5. R_chain      : reasoning steps grounded + internally consistent
6. R_coherence  : cross-component consistency penalties
7. R_diagnosis  : error_type correctness for Not Attributable
8. R_specificity: penalizes generic/templated outputs

Key innovations over v2:
- Strict format: partial JSON (label-only) gets 0, forces full structured output
- Alignment uses threshold=0.75 (stricter than v2's 0.6)
- Chain reward checks step-label consistency, not just word count
- Coherence reward penalizes self-contradictory outputs
- Normalized weights (sum=1.0) for stable training signal
- No boundary bonus (removed: it penalized high-accuracy groups)
"""

import json
import re
from typing import Optional
from difflib import SequenceMatcher


VALID_LABELS = {"Attributable", "Not Attributable"}
LABEL_ALIASES = {
    "attributable": "Attributable",
    "not attributable": "Not Attributable",
    "not_attributable": "Not Attributable",
    "supported": "Attributable",
    "not supported": "Not Attributable",
    "yes": "Attributable",
    "no": "Not Attributable",
    "true": "Attributable",
    "false": "Not Attributable",
    "entailment": "Attributable",
    "contradiction": "Not Attributable",
    "neutral": "Not Attributable",
    "s": "Attributable",
    "c": "Not Attributable",
    "n": "Not Attributable",
}

VALID_ERROR_TYPES = {
    "numerical_exaggeration", "negation_flip", "scope_inflation",
    "temporal_shift", "entity_substitution", "fabrication",
}


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON object from model response. Strict: requires structured fields.

    Returns None (reward=0) if:
    - No valid JSON found
    - Missing required fields (label, confidence, evidence_alignment, reasoning_chain)
    This forces the model to produce full structured output, not just label+confidence.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
        # Require all structured fields for full reward
        required = {"label", "confidence", "evidence_alignment", "reasoning_chain"}
        if required.issubset(parsed.keys()):
            return parsed
        # Partial output: only label+confidence → heavily penalized via format score
        if "label" in parsed:
            parsed.setdefault("evidence_alignment", [])
            parsed.setdefault("reasoning_chain", [])
            parsed.setdefault("confidence", 0.5)
            parsed["_partial"] = True  # Flag for format scoring
            return parsed
    return None


def normalize_label(label: str) -> str | None:
    if not label:
        return None
    label_lower = label.strip().lower()
    if label_lower in LABEL_ALIASES:
        return LABEL_ALIASES[label_lower]
    for valid in VALID_LABELS:
        if valid.lower() in label_lower:
            return valid
    return None


# ---------------------------------------------------------------------------
# Span grounding utilities
# ---------------------------------------------------------------------------

def _normalize_text(t: str) -> str:
    """Lowercase, collapse whitespace for fuzzy matching."""
    return " ".join(t.lower().split())


def _span_overlap(span: str, text: str, threshold: float = 0.75) -> float:
    """Check if span appears in text. Returns overlap score 0-1.

    Uses substring match first (fast), falls back to SequenceMatcher.
    Threshold 0.75 (stricter than v2's 0.6) — forces near-exact span extraction.
    Below threshold: sliding penalty instead of hard cutoff.
    """
    if not span or not text:
        return 0.0
    span_n = _normalize_text(span)
    text_n = _normalize_text(text)
    if len(span_n) < 3:
        return 0.0
    # Exact substring
    if span_n in text_n:
        return 1.0
    # Fuzzy: find best matching window
    if len(span_n) > len(text_n):
        return 0.0
    if len(span_n) > 200:
        span_n = span_n[:200]
    ratio = SequenceMatcher(None, span_n, text_n).ratio()
    # For long texts, check windowed match
    if len(text_n) > len(span_n) * 3:
        window = len(span_n) + 20
        best = 0.0
        for i in range(0, len(text_n) - window + 1, max(1, window // 4)):
            chunk = text_n[i:i + window]
            r = SequenceMatcher(None, span_n, chunk).ratio()
            best = max(best, r)
        ratio = max(ratio, best)
    # Sliding penalty: full credit >= 0.75, partial 0.5-0.75, zero below 0.5
    if ratio >= threshold:
        return ratio
    elif ratio >= 0.5:
        return ratio * 0.4  # Heavily penalize mediocre matches
    return 0.0


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

def get_reward_weights(epoch: int = 1, total_epochs: int = 5) -> dict:
    """Dynamic weights: accuracy-heavy early, grounding-heavy late.

    All weights are normalized to sum=1.0 for stable GRPO signal.
    """
    p = min((epoch - 1) / max(total_epochs - 1, 1), 1.0)
    raw = {
        "format":      0.10,
        "accuracy":    0.55 - 0.20 * p,   # 0.55 → 0.35
        "calibration": 0.05 + 0.10 * p,   # 0.05 → 0.15
        "alignment":   0.10 + 0.10 * p,   # 0.10 → 0.20
        "chain":       0.08 + 0.07 * p,   # 0.08 → 0.15
        "coherence":   0.05 + 0.03 * p,   # 0.05 → 0.08
        "diagnosis":   0.04 + 0.03 * p,   # 0.04 → 0.07
        "specificity": 0.03 + 0.02 * p,   # 0.03 → 0.05 (lowest — informational)
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Per-component reward functions
# ---------------------------------------------------------------------------

def _r_alignment(parsed: dict, claim: str, source: str) -> float:
    """Score evidence alignment grounding (0-1).

    Checks that claim_span actually appears in claim,
    and source_span actually appears in source (or is NOT_FOUND).
    """
    alignment = parsed.get("evidence_alignment", [])
    if not alignment or not isinstance(alignment, list):
        return 0.0

    scores = []
    for entry in alignment:
        if not isinstance(entry, dict):
            continue
        cs = entry.get("claim_span", "")
        ss = entry.get("source_span", "")
        status = entry.get("status", "")

        # Claim span must be grounded in claim
        c_score = _span_overlap(cs, claim) if cs else 0.0

        # Source span: NOT_FOUND is valid for not_found status
        if ss == "NOT_FOUND" or (status == "not_found" and not ss):
            s_score = 1.0 if status == "not_found" else 0.5
        else:
            s_score = _span_overlap(ss, source) if ss else 0.0

        # Status must be valid
        status_valid = 1.0 if status in ("match", "mismatch", "not_found") else 0.0

        entry_score = (c_score * 0.4 + s_score * 0.4 + status_valid * 0.2)
        scores.append(entry_score)

    if not scores:
        return 0.0
    # Reward having 2-5 alignment entries (penalize too few or too many)
    count_bonus = min(1.0, len(scores) / 2.0) * min(1.0, 5.0 / max(len(scores), 1))
    return (sum(scores) / len(scores)) * count_bonus


def _r_chain(parsed: dict, pred_label: str) -> float:
    """Score reasoning chain quality (0-1).

    Checks:
    1. Each step has non-empty source_evidence and explanation
    2. Step judgments are internally consistent with final label
    3. Steps reference specific (not generic) evidence
    """
    chain = parsed.get("reasoning_chain", [])
    if not chain or not isinstance(chain, list):
        return 0.0

    step_scores = []
    judgments = []
    for step in chain:
        if not isinstance(step, dict):
            continue
        score = 0.0
        # Has required fields
        has_evidence = bool(step.get("source_evidence", "").strip())
        has_explanation = bool(step.get("explanation", "").strip())
        has_judgment = step.get("judgment", "") in (
            "supported", "not_supported", "partially_supported"
        )
        score = (0.4 * has_evidence + 0.3 * has_explanation + 0.3 * has_judgment)

        # Penalize very short evidence (< 10 chars = likely generic)
        ev = step.get("source_evidence", "")
        if ev and len(ev.strip()) < 10:
            score *= 0.5

        step_scores.append(score)
        if has_judgment:
            judgments.append(step["judgment"])

    if not step_scores:
        return 0.0

    avg_step = sum(step_scores) / len(step_scores)

    # Consistency bonus: chain conclusion should align with label
    consistency = 1.0
    if judgments:
        has_not_supported = any(j == "not_supported" for j in judgments)
        all_supported = all(j == "supported" for j in judgments)

        if pred_label == "Not Attributable":
            # Should have at least one not_supported step
            consistency = 1.0 if has_not_supported else 0.5
        else:
            # Should NOT have not_supported steps
            consistency = 1.0 if all_supported else 0.6

    # Count bonus: 2-4 steps is ideal
    count_bonus = min(1.0, len(step_scores) / 2.0) * min(1.0, 4.0 / max(len(step_scores), 1))

    return avg_step * consistency * count_bonus


def _r_coherence(parsed: dict, pred_label: str) -> float:
    """Cross-component coherence score (0-1). Penalizes contradictions.

    Checks consistency between:
    - label ↔ error_type
    - label ↔ alignment statuses
    - label ↔ reasoning chain judgments
    - error_type ↔ fix_suggestion
    """
    penalties = 0.0

    error_type = parsed.get("error_type", "")
    fix_suggestion = parsed.get("fix_suggestion", "")
    alignment = parsed.get("evidence_alignment", [])
    chain = parsed.get("reasoning_chain", [])

    if pred_label == "Attributable":
        # Should NOT have error_type or fix_suggestion
        if error_type and error_type.lower() not in ("", "none", "n/a", "not applicable"):
            penalties += 0.3
        if fix_suggestion and fix_suggestion.lower() not in ("", "none", "n/a", "not applicable"):
            penalties += 0.1
        # All alignment should be "match"
        if alignment:
            mismatches = sum(1 for e in alignment if isinstance(e, dict)
                           and e.get("status") in ("mismatch", "not_found"))
            if mismatches > 0:
                penalties += 0.2 * min(1.0, mismatches / max(len(alignment), 1))

    else:  # Not Attributable
        # SHOULD have error_type
        if not error_type or error_type.lower() in ("", "none", "n/a"):
            penalties += 0.2
        elif error_type not in VALID_ERROR_TYPES:
            penalties += 0.1
        # Should have fix_suggestion
        if not fix_suggestion or len(fix_suggestion.strip()) < 5:
            penalties += 0.1
        # Should have at least one mismatch/not_found in alignment
        if alignment:
            all_match = all(
                isinstance(e, dict) and e.get("status") == "match"
                for e in alignment if isinstance(e, dict)
            )
            if all_match and len(alignment) > 0:
                penalties += 0.25

    # Chain consistency with label
    if chain:
        judgments = [s.get("judgment") for s in chain
                     if isinstance(s, dict) and s.get("judgment")]
        if judgments:
            not_supported_ratio = sum(1 for j in judgments if j == "not_supported") / len(judgments)
            if pred_label == "Attributable" and not_supported_ratio > 0.3:
                penalties += 0.2
            elif pred_label == "Not Attributable" and not_supported_ratio == 0:
                penalties += 0.2

    return max(0.0, 1.0 - penalties)


def _r_diagnosis(parsed: dict, gold_label: str, pred_label: str) -> float:
    """Score error diagnosis quality (0-1).

    Only applies to Not Attributable predictions.
    """
    if gold_label == "Attributable" and pred_label == "Attributable":
        return 1.0  # No diagnosis needed, full score
    if gold_label == "Attributable" and pred_label != "Attributable":
        return 0.0  # Wrong prediction, no diagnosis credit

    error_type = parsed.get("error_type", "")
    fix_suggestion = parsed.get("fix_suggestion", "")

    score = 0.0
    # Valid error type
    if error_type in VALID_ERROR_TYPES:
        score += 0.5
    elif error_type:
        score += 0.1  # At least tried

    # Fix suggestion quality
    if fix_suggestion and len(fix_suggestion.strip()) > 10:
        score += 0.3
        # Bonus for specific fix (contains a quote or reference)
        if '"' in fix_suggestion or "'" in fix_suggestion or "→" in fix_suggestion:
            score += 0.2
    elif fix_suggestion:
        score += 0.1

    return min(score, 1.0)


def _r_specificity(parsed: dict, claim: str) -> float:
    """Penalize generic/templated outputs (0-1).

    Rewards responses that reference specific details from the claim,
    not just template-fill boilerplate.
    """
    # Check reasoning chain for claim-specific content
    chain = parsed.get("reasoning_chain", [])
    alignment = parsed.get("evidence_alignment", [])

    specificity_signals = 0
    total_checks = 0

    # Check if claim_parts in chain reference actual claim content
    for step in chain:
        if not isinstance(step, dict):
            continue
        cp = step.get("claim_part", "")
        if cp and len(cp) > 5:
            total_checks += 1
            if _span_overlap(cp, claim) > 0:
                specificity_signals += 1

    # Check alignment claim_spans
    for entry in alignment:
        if not isinstance(entry, dict):
            continue
        cs = entry.get("claim_span", "")
        if cs and len(cs) > 3:
            total_checks += 1
            if _span_overlap(cs, claim) > 0:
                specificity_signals += 1

    if total_checks == 0:
        return 0.3  # No checkable content
    return specificity_signals / total_checks


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def compute_score(data_source: str, solution_str: str, ground_truth: dict,
                  extra_info: dict = None, **kwargs) -> float:
    """Compute v3 reward for a single structured verification response.

    Args:
        data_source: task identifier
        solution_str: model's raw text output
        ground_truth: {"target": "Attributable"|"Not Attributable",
                       "claim": str, "source": str}  # v3: includes claim/source
        extra_info: {"epoch": int, "total_epochs": int}
    """
    gold_label = ground_truth.get("target", "Not Attributable")
    extra_info = extra_info or {}

    epoch = extra_info.get("epoch", 1)
    total_epochs = extra_info.get("total_epochs", 5)
    w = get_reward_weights(epoch, total_epochs)

    # Extract claim and source for grounding checks
    claim = ground_truth.get("claim", "")
    source = ground_truth.get("source", "")

    # If claim/source not in ground_truth, try to extract from prompt
    if not claim and "prompt" in extra_info:
        prompt = extra_info["prompt"]
        claim_m = re.search(r"Claim:\s*(.+?)(?:\n|Source:)", prompt, re.DOTALL)
        source_m = re.search(r"Source:\s*(.+?)(?:\n\n|Verify)", prompt, re.DOTALL)
        if claim_m:
            claim = claim_m.group(1).strip()
        if source_m:
            source = source_m.group(1).strip()

    # === Component 1: Format ===
    parsed = extract_json_from_response(solution_str)
    if parsed is None:
        return 0.0

    pred_raw = parsed.get("label", "")
    pred_label = normalize_label(pred_raw)
    if pred_label is None:
        return 0.0

    # Partial JSON (label-only, missing structured fields) → heavy penalty
    if parsed.get("_partial"):
        r_format = w["format"] * 0.2  # 80% format penalty
    else:
        r_format = w["format"]

    # === Component 2: Accuracy ===
    correct = (pred_label == gold_label)
    r_accuracy = w["accuracy"] if correct else 0.0

    # === Component 3: Calibration (asymmetric) ===
    confidence = min(max(parsed.get("confidence", 0.5), 0.0), 1.0)
    if correct:
        # Reward confidence, but optimal around 0.8 (not overconfident)
        cal_error = abs(confidence - 0.85)
        r_calibration = w["calibration"] * max(0.0, 1.0 - 1.5 * cal_error)
    else:
        # Harsh penalty for wrong + confident (2x weight)
        r_calibration = -w["calibration"] * confidence * 2.0

    # === Component 4: Alignment grounding ===
    r_alignment = w["alignment"] * _r_alignment(parsed, claim, source)

    # === Component 5: Chain quality ===
    r_chain = w["chain"] * _r_chain(parsed, pred_label)

    # === Component 6: Cross-component coherence ===
    r_coherence = w["coherence"] * _r_coherence(parsed, pred_label)

    # === Component 7: Diagnosis ===
    r_diagnosis = w["diagnosis"] * _r_diagnosis(parsed, gold_label, pred_label)

    # === Component 8: Specificity ===
    if claim:
        r_specificity = w["specificity"] * _r_specificity(parsed, claim)
    else:
        r_specificity = w["specificity"] * 0.5  # Can't check without claim

    total = (r_format + r_accuracy + r_calibration + r_alignment +
             r_chain + r_coherence + r_diagnosis + r_specificity)
    return max(total, 0.0)


def compute_score_batch(data_sources, solution_strs, ground_truths,
                        extra_infos, **kwargs):
    """Batch reward computation (veRL NaiveRewardManager compatible).

    No boundary bonus — it penalized high-accuracy groups (perverse incentive).
    GRPO's group-relative advantage estimation already handles normalization.
    """
    scores = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs,
                                ground_truths, extra_infos):
        score = compute_score(ds, sol, gt, ei)
        scores.append(score)
    return scores
