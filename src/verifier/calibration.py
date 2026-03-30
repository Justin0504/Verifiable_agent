"""Verifier calibration via synthetic contrastive data.

Generates claim-evidence pairs with known ground truth labels (S/C/N)
by corrupting facts from the knowledge base. Runs the verifier on these
pairs to measure accuracy and detect systematic biases.
"""

from __future__ import annotations

import random
import re
import uuid

from src.data.schema import AtomicClaim, ClaimLabel

from .knowledge_base import Document, KnowledgeBase


def generate_calibration_set(
    kb: KnowledgeBase,
    n_per_label: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic claim-evidence pairs with known labels.

    Strategy:
    - Supported: extract a fact directly from evidence (paraphrased)
    - Contradicted: take a fact and corrupt a key detail (number, name, date)
    - Not Mentioned: pair a claim with unrelated evidence

    Returns list of {claim, evidence, gold_label, source_doc_id, corruption_type}.
    """
    rng = random.Random(seed)
    docs = kb.documents
    if len(docs) < 3:
        return []

    calibration_data: list[dict] = []

    # ── Supported: claim matches evidence ─────────────────────────
    sampled = rng.sample(docs, min(n_per_label, len(docs)))
    for doc in sampled:
        sentences = _split_sentences(doc.content)
        if not sentences:
            continue
        sent = rng.choice(sentences)
        calibration_data.append({
            "id": str(uuid.uuid4()),
            "claim": sent,
            "evidence": doc.content,
            "gold_label": "S",
            "source_doc_id": doc.id,
            "corruption_type": None,
        })

    # ── Contradicted: corrupt a fact in the claim ─────────────────
    sampled = rng.sample(docs, min(n_per_label, len(docs)))
    for doc in sampled:
        sentences = _split_sentences(doc.content)
        if not sentences:
            continue
        sent = rng.choice(sentences)
        corrupted, ctype = _corrupt_sentence(sent, rng)
        if corrupted and corrupted != sent:
            calibration_data.append({
                "id": str(uuid.uuid4()),
                "claim": corrupted,
                "evidence": doc.content,
                "gold_label": "C",
                "source_doc_id": doc.id,
                "corruption_type": ctype,
            })

    # ── Not Mentioned: claim from doc A, evidence from doc B ──────
    if len(docs) >= 2:
        for _ in range(min(n_per_label, len(docs))):
            doc_a, doc_b = rng.sample(docs, 2)
            sentences_a = _split_sentences(doc_a.content)
            if not sentences_a:
                continue
            claim = rng.choice(sentences_a)
            calibration_data.append({
                "id": str(uuid.uuid4()),
                "claim": claim,
                "evidence": doc_b.content,
                "gold_label": "N",
                "source_doc_id": doc_a.id,
                "corruption_type": None,
            })

    rng.shuffle(calibration_data)
    return calibration_data


def evaluate_verifier_accuracy(
    predictions: list[dict],
) -> dict:
    """Compare verifier predictions against gold labels.

    Each item in predictions should have: gold_label, predicted_label.

    Returns accuracy, per-label precision/recall, confusion matrix,
    and detected biases.
    """
    if not predictions:
        return {"accuracy": 0.0}

    total = len(predictions)
    correct = sum(1 for p in predictions if p["gold_label"] == p["predicted_label"])
    accuracy = correct / total

    # Per-label stats
    labels = ["S", "C", "N"]
    per_label: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for p in predictions if p["gold_label"] == label and p["predicted_label"] == label)
        fp = sum(1 for p in predictions if p["gold_label"] != label and p["predicted_label"] == label)
        fn = sum(1 for p in predictions if p["gold_label"] == label and p["predicted_label"] != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for p in predictions:
        g = p["gold_label"]
        pred = p["predicted_label"]
        if g in confusion and pred in confusion[g]:
            confusion[g][pred] += 1

    # Detect systematic biases
    biases: list[str] = []
    for label in labels:
        if per_label[label]["precision"] < 0.5 and per_label[label]["fp"] > 3:
            biases.append(f"Over-predicts {label}: precision={per_label[label]['precision']:.2f}")
        if per_label[label]["recall"] < 0.5 and per_label[label]["fn"] > 3:
            biases.append(f"Under-predicts {label}: recall={per_label[label]['recall']:.2f}")

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "biases": biases,
    }


def bias_to_prompt_correction(biases: list[str]) -> str:
    """Convert detected biases into a prompt correction string.

    This is injected into the verifier's system prompt to counteract
    systematic errors discovered during calibration.
    """
    if not biases:
        return ""

    lines = ["\n\nCALIBRATION CORRECTIONS (based on measured systematic errors):"]
    for bias in biases:
        if "Over-predicts S" in bias:
            lines.append(
                "- You tend to label claims as Supported too easily. "
                "Require stronger evidence before assigning S."
            )
        elif "Over-predicts C" in bias:
            lines.append(
                "- You tend to label claims as Contradicted too aggressively. "
                "Only assign C when the evidence DIRECTLY conflicts."
            )
        elif "Over-predicts N" in bias:
            lines.append(
                "- You default to Not Mentioned too often. "
                "Look more carefully for implicit support or contradiction."
            )
        elif "Under-predicts S" in bias:
            lines.append(
                "- You miss supported claims. Accept paraphrased or "
                "indirectly stated evidence as support."
            )
        elif "Under-predicts C" in bias:
            lines.append(
                "- You miss contradictions. Pay close attention to "
                "numbers, dates, and names that differ from the evidence."
            )
        elif "Under-predicts N" in bias:
            lines.append(
                "- You rarely assign Not Mentioned. If the evidence truly "
                "doesn't address the claim, label it N."
            )
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering short ones."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _corrupt_sentence(sentence: str, rng: random.Random) -> tuple[str, str]:
    """Corrupt a sentence by changing a number, date, or name.

    Returns (corrupted_sentence, corruption_type).
    """
    # Try number corruption first
    numbers = re.findall(r'\b\d{3,}\b', sentence)
    if numbers:
        target = rng.choice(numbers)
        n = int(target)
        offset = rng.choice([-3, -2, -1, 1, 2, 3])
        corrupted = sentence.replace(target, str(n + offset), 1)
        return corrupted, "number"

    # Try year corruption
    years = re.findall(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', sentence)
    if years:
        target = rng.choice(years)
        y = int(target)
        offset = rng.choice([-5, -3, -2, 2, 3, 5])
        corrupted = sentence.replace(target, str(y + offset), 1)
        return corrupted, "year"

    # Try swapping a capitalized word (likely a name)
    names = re.findall(r'\b[A-Z][a-z]{2,}\b', sentence)
    replacements = ["Smith", "Johnson", "Berlin", "Tokyo", "Newton", "Darwin"]
    if len(names) >= 1:
        target = rng.choice(names)
        candidates = [r for r in replacements if r != target]
        if candidates:
            corrupted = sentence.replace(target, rng.choice(candidates), 1)
            return corrupted, "name"

    return sentence, "none"
