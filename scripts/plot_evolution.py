# -*- coding: utf-8 -*-
"""Plot self-evolution curves and results from v9 experiment."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def load_data(output_dir):
    curve_path = Path(output_dir) / "evolution_curve.json"
    comp_path = Path(output_dir) / "comparison.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
    comp = json.loads(comp_path.read_text()) if comp_path.exists() else {}
    return curve, comp


def plot_evolution_curve(curve, comp, output_dir):
    """Plot 1: Calibration accuracy over epochs with annotations."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Extract data
    epochs = [0]  # epoch 0 baseline
    cal_accs = [comp.get("baseline_cal_accuracy", 0)]

    for entry in curve:
        if entry.get("skip"):
            continue
        epochs.append(entry["epoch"])
        cal_accs.append(entry.get("cal_accuracy", cal_accs[-1]))

    # Main line
    ax.plot(epochs, [a * 100 for a in cal_accs], "o-", color="#1565C0",
            linewidth=2.5, markersize=8, label="Verifier Agent (self-evolution)", zorder=5)

    # Shade improvement region
    baseline = cal_accs[0] * 100
    ax.axhline(y=baseline, color="#999", linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between(epochs, baseline, [a * 100 for a in cal_accs],
                     alpha=0.12, color="#1565C0")

    # Annotate key points
    ax.annotate(f"Baseline: {baseline:.1f}%",
                xy=(0, baseline), xytext=(1.5, baseline - 4),
                fontsize=10, color="#666",
                arrowprops=dict(arrowstyle="->", color="#999"))

    best_acc = max(cal_accs) * 100
    best_epoch = epochs[cal_accs.index(max(cal_accs))]
    ax.annotate(f"Best: {best_acc:.1f}% (+{best_acc - baseline:.1f}pp)",
                xy=(best_epoch, best_acc), xytext=(best_epoch + 1, best_acc + 2),
                fontsize=11, fontweight="bold", color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Calibration Accuracy (%)")
    ax.set_title("Self-Evolution Curve: Verifier Accuracy Over Training Epochs")
    ax.set_xlim(-0.5, max(epochs) + 1.5)
    ax.set_ylim(min(cal_accs) * 100 - 5, best_acc + 8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "evolution_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_baseline_comparison(comp, output_dir):
    """Plot 2: Bar chart comparing verifier agent vs baselines."""
    metrics = comp.get("metrics", {})
    if not metrics:
        print("  No metrics found, skipping baseline comparison plot.")
        return

    # Sort by accuracy
    sorted_methods = sorted(metrics.items(), key=lambda x: x[1]["accuracy"])
    names = [n.replace("_", " ").title() for n, _ in sorted_methods]
    accs = [m["accuracy"] * 100 for _, m in sorted_methods]
    f1s = [m.get("macro_f1", 0) for _, m in sorted_methods]

    # Color: verifier agent in blue, others in gray
    colors = []
    for n, _ in sorted_methods:
        if n == "verifier_agent":
            colors.append("#1565C0")
        else:
            colors.append("#B0BEC5")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, acc, f1 in zip(bars, accs, f1s):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}% (F1={f1:.3f})",
                va="center", fontsize=10, fontweight="bold" if bar.get_facecolor()[:3] != (0.69, 0.75, 0.77) else "normal")

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Verifier Agent vs. Baselines on TruthfulQA")
    ax.set_xlim(0, max(accs) + 12)
    ax.grid(True, axis="x", alpha=0.3)

    out = Path(output_dir) / "baseline_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_training_details(curve, output_dir):
    """Plot 3: Multi-panel — synth accuracy, data size, rules, prompt length."""
    valid = [e for e in curve if not e.get("skip")]
    if len(valid) < 2:
        return

    epochs = [e["epoch"] for e in valid]
    synth_acc = [e.get("synth_accuracy", 0) * 100 for e in valid]
    synth_n = [e.get("synth_n", 0) for e in valid]
    rules = [e.get("rb_rules", 0) for e in valid]
    prompt_len = [e.get("prompt_len", 0) for e in valid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Self-Evolution Training Details", fontsize=14, fontweight="bold")

    # Synth accuracy
    axes[0, 0].plot(epochs, synth_acc, "s-", color="#E65100", markersize=6)
    axes[0, 0].set_ylabel("Synth Train Acc (%)")
    axes[0, 0].set_title("Training Accuracy on Synthetic Data")
    axes[0, 0].grid(True, alpha=0.3)

    # Data size per epoch
    axes[0, 1].bar(epochs, synth_n, color="#43A047", alpha=0.8)
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].set_title("Training Samples per Epoch (after filtering)")
    axes[0, 1].grid(True, alpha=0.3)

    # ReasoningBank rules
    axes[1, 0].plot(epochs, rules, "D-", color="#7B1FA2", markersize=6)
    axes[1, 0].set_ylabel("Total Rules")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_title("ReasoningBank Rules Accumulated")
    axes[1, 0].grid(True, alpha=0.3)

    # Prompt length
    axes[1, 1].plot(epochs, prompt_len, "^-", color="#00838F", markersize=6)
    axes[1, 1].set_ylabel("Characters")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_title("Evolved Prompt Length")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "training_details.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "results/fair_comparison_truthfulqa_v9"
    print(f"Loading from: {output_dir}")

    curve, comp = load_data(output_dir)
    if not curve:
        print("No evolution curve data yet. Wait for experiment to finish.")
        return

    print(f"Found {len(curve)} epochs")
    plot_evolution_curve(curve, comp, output_dir)
    plot_baseline_comparison(comp, output_dir)
    plot_training_details(curve, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
