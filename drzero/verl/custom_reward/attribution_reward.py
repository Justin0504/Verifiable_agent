"""Reward function for fact attribution GRPO training (SEVA).

Task: Given (claim, source) → Attributable / Not Attributable

Reward components:
1. R_format    (w_f): valid JSON with label ∈ {Attributable, Not Attributable}
2. R_accuracy  (w_a): correct label
3. R_calibration (w_c): confidence alignment with correctness
4. R_reasoning (w_r): non-trivial explanation
5. R_rule_cite (w_rc): cited applicable ReasoningBank rules

Two post-hoc adjustments (applied in compute_score_batch):
- R_boundary: group-level boundary-optimal weighting (Dr.Zero)
  = 1 - |mean(correct_in_group) - 0.5| * 2
- Dynamic weights: shift from accuracy→calibration+reasoning over epochs
"""

import json
import re
from typing import Optional


VALID_LABELS = {"Attributable", "Not Attributable"}
# Shortcuts we accept and normalize
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
    # S/C/N from old format
    "s": "Attributable",
    "c": "Not Attributable",
    "n": "Not Attributable",
}


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON object from model response, with robust fallbacks."""
    text = text.strip()

    # Try direct JSON parse
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction
    label_match = re.search(
        r'"label"\s*:\s*"([^"]+)"', text, re.IGNORECASE
    )
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    rules_match = re.search(r'"rules_cited"\s*:\s*\[([^\]]*)\]', text)

    if label_match:
        rules = []
        if rules_match:
            rules = [r.strip().strip('"') for r in rules_match.group(1).split(",") if r.strip()]
        return {
            "label": label_match.group(1),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reasoning": reason_match.group(1) if reason_match else "",
            "rules_cited": rules,
        }
    return None


def normalize_label(label: str) -> str | None:
    """Normalize predicted label to canonical form."""
    if not label:
        return None
    label_lower = label.strip().lower()
    if label_lower in LABEL_ALIASES:
        return LABEL_ALIASES[label_lower]
    # Check if any valid label is a substring
    for valid in VALID_LABELS:
        if valid.lower() in label_lower:
            return valid
    return None


def get_reward_weights(epoch: int = 1, total_epochs: int = 5) -> dict:
    """Dynamic reward weights: accuracy-heavy early, calibration-heavy late.

    Epoch 1: focus on getting the label right.
    Epoch 5: focus on calibration, reasoning quality, and rule usage.
    """
    progress = min((epoch - 1) / max(total_epochs - 1, 1), 1.0)
    return {
        "format":    0.1,
        "accuracy":  1.0 - 0.3 * progress,      # 1.0 → 0.7
        "calibration": 0.2 + 0.3 * progress,    # 0.2 → 0.5
        "reasoning": 0.1 + 0.15 * progress,     # 0.1 → 0.25
        "rule_cite": 0.05 + 0.1 * progress,     # 0.05 → 0.15
    }


def compute_score(data_source: str, solution_str: str, ground_truth: dict,
                  extra_info: dict = None, **kwargs) -> float:
    """Compute reward for a single attribution verification response.

    Args:
        data_source: task identifier (e.g., "attribution")
        solution_str: model's raw text output
        ground_truth: {"target": "Attributable" or "Not Attributable"}
        extra_info: optional metadata, may contain "epoch" and "total_epochs"

    Returns:
        float: scalar reward
    """
    gold_label = ground_truth.get("target", "Not Attributable")
    extra_info = extra_info or {}

    # Dynamic weights based on training epoch
    epoch = extra_info.get("epoch", 1)
    total_epochs = extra_info.get("total_epochs", 5)
    w = get_reward_weights(epoch, total_epochs)

    # === Component 1: Format reward ===
    parsed = extract_json_from_response(solution_str)
    if parsed is None:
        return 0.0

    pred_raw = parsed.get("label", "")
    pred_label = normalize_label(pred_raw)
    if pred_label is None:
        return 0.0  # Invalid label → zero reward

    r_format = w["format"]

    confidence = min(max(parsed.get("confidence", 0.5), 0.0), 1.0)
    reasoning = parsed.get("reasoning", "")
    rules_cited = parsed.get("rules_cited", [])

    # === Component 2: Accuracy reward ===
    correct = (pred_label == gold_label)
    r_accuracy = w["accuracy"] if correct else 0.0

    # === Component 3: Calibration bonus ===
    if correct:
        r_calibration = w["calibration"] * confidence
    else:
        r_calibration = -w["calibration"] * confidence

    # === Component 4: Reasoning bonus ===
    r_reasoning = 0.0
    if reasoning and len(reasoning.split()) >= 5:
        r_reasoning = w["reasoning"]

    # === Component 5: Rule citation bonus ===
    r_rule_cite = 0.0
    if rules_cited and correct:
        r_rule_cite = w["rule_cite"]

    total = r_format + r_accuracy + r_calibration + r_reasoning + r_rule_cite
    return max(total, 0.0)


def apply_boundary_bonus(scores: list[float], group_size: int = 5) -> list[float]:
    """Group-level boundary-optimal weighting (Dr.Zero).

    Amplifies reward for samples where the model gets ~50% correct within
    the GRPO group (maximum learning signal at decision boundary).
    Attenuates reward for trivially easy or impossible samples.

    Applied as a post-hoc pass AFTER per-sample compute_score.
    """
    if len(scores) < group_size:
        return scores

    bonused = []
    for i in range(0, len(scores), group_size):
        group = scores[i:i + group_size]
        # Correct = got a non-trivial reward (accuracy component fired)
        correct_count = sum(1 for s in group if s > 0.5)
        correct_rate = correct_count / len(group)
        # Boundary bonus: max at 50%, zero at 0% or 100%
        boundary = 1.0 - abs(correct_rate - 0.5) * 2.0
        # Scale: α=0.5 baseline + β=0.5 * boundary
        # So easy/hard samples still get 50% of their reward (not zeroed out)
        scale = 0.5 + 0.5 * boundary
        for s in group:
            bonused.append(s * scale)

    # Handle remainder (partial group at end)
    remainder = len(scores) % group_size
    if remainder > 0:
        bonused.extend(scores[-remainder:])

    return bonused


def compute_score_batch(data_sources, solution_strs, ground_truths,
                        extra_infos, **kwargs):
    """Batch reward computation (called by veRL's NaiveRewardManager).

    Applies per-sample structured reward, then group-level boundary bonus.
    """
    scores = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs,
                                ground_truths, extra_infos):
        score = compute_score(ds, sol, gt, ei)
        scores.append(score)

    # Apply boundary-optimal weighting across GRPO groups
    group_size = kwargs.get("group_size", 5)
    scores = apply_boundary_bonus(scores, group_size=group_size)

    return scores
