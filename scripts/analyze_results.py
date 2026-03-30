"""Analyze experiment results and generate tables + figures.

Usage:
    python scripts/analyze_results.py --results results/default_20250328_120000/all_results.json
    python scripts/analyze_results.py --results-dir results/default_20250328_120000/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_model_risk_table(all_results: dict) -> pd.DataFrame:
    """Table 1: Hallucination rate per model per risk type."""
    rows = []
    for model_name, epochs in all_results.items():
        # Use last epoch metrics
        last_epoch = epochs[-1]
        metrics = last_epoch["metrics"]
        per_risk = metrics.get("per_risk_type", {})

        for risk_type, data in per_risk.items():
            rows.append({
                "Model": model_name,
                "Risk Type": risk_type,
                "Hallucination Rate": data["hallucination_rate"],
                "Avg Score": data["avg_score"],
                "S Ratio": data["s_ratio"],
                "C Ratio": data["c_ratio"],
                "N Ratio": data["n_ratio"],
                "Count": data["count"],
            })

    return pd.DataFrame(rows)


def build_scn_table(all_results: dict) -> pd.DataFrame:
    """Table 2: S/C/N distribution per model."""
    rows = []
    for model_name, epochs in all_results.items():
        last_epoch = epochs[-1]
        m = last_epoch["metrics"]
        rows.append({
            "Model": model_name,
            "Total Claims": m["total_claims"],
            "Supported (S)": f"{m['s_ratio']:.1%}",
            "Contradicted (C)": f"{m['c_ratio']:.1%}",
            "Not Mentioned (N)": f"{m['n_ratio']:.1%}",
            "Hallucination Rate": f"{m['hallucination_rate']:.1%}",
            "Avg Score": f"{m['avg_score']:.3f}",
        })

    return pd.DataFrame(rows)


def plot_radar_chart(all_results: dict, output_path: str) -> None:
    """Figure 1: Radar chart comparing models across risk types."""
    risk_types = ["missing_evidence", "multi_hop", "pressure_presupposition", "unanswerable"]
    labels = ["Missing\nEvidence", "Multi-Hop", "Pressure/\nPresupposition", "Unanswerable"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(risk_types), endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    for idx, (model_name, epochs) in enumerate(all_results.items()):
        last_epoch = epochs[-1]
        per_risk = last_epoch["metrics"].get("per_risk_type", {})
        # Use 1 - hallucination_rate as "reliability"
        values = [1 - per_risk.get(rt, {}).get("hallucination_rate", 0) for rt in risk_types]
        values += values[:1]

        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        ax.plot(angles, values, "o-", linewidth=2, label=short_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.set_title("Model Reliability by Risk Type\n(1 - Hallucination Rate)", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved radar chart to {output_path}")


def plot_evolution_curve(all_results: dict, output_path: str) -> None:
    """Figure 2: Hallucination rate across evolution epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    for idx, (model_name, epochs) in enumerate(all_results.items()):
        epoch_nums = [e["epoch"] + 1 for e in epochs]
        hall_rates = [e["metrics"]["hallucination_rate"] for e in epochs]
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        ax.plot(epoch_nums, hall_rates, "o-", linewidth=2, markersize=8,
                label=short_name, color=colors[idx])

    ax.set_xlabel("Evolution Epoch", fontsize=12)
    ax.set_ylabel("Hallucination Rate", fontsize=12)
    ax.set_title("Hallucination Rate Across Self-Evolution Epochs", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved evolution curve to {output_path}")


def plot_scn_heatmap(all_results: dict, output_path: str) -> None:
    """Figure 3: S/C/N distribution heatmap per model per risk type."""
    risk_types = ["missing_evidence", "multi_hop", "pressure_presupposition", "unanswerable"]

    models = []
    c_ratios = []
    for model_name, epochs in all_results.items():
        last_epoch = epochs[-1]
        per_risk = last_epoch["metrics"].get("per_risk_type", {})
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        models.append(short_name)
        c_ratios.append([per_risk.get(rt, {}).get("c_ratio", 0) for rt in risk_types])

    df = pd.DataFrame(c_ratios, index=models,
                      columns=["Missing Evidence", "Multi-Hop", "Pressure/Presup.", "Unanswerable"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.8)))
    sns.heatmap(df, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("Contradiction Rate (C) by Model × Risk Type", fontsize=14)
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results", required=True, help="Path to all_results.json")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output) if args.output else results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_results(str(results_path))

    # Table 1: Model × Risk Type hallucination rates
    print("\n" + "=" * 80)
    print("TABLE 1: Hallucination Rate per Model per Risk Type")
    print("=" * 80)
    df1 = build_model_risk_table(all_results)
    if not df1.empty:
        pivot = df1.pivot_table(
            index="Model", columns="Risk Type", values="Hallucination Rate"
        )
        print(pivot.to_string(float_format="%.1%%"))
        pivot.to_csv(str(output_dir / "table1_hallucination_rate.csv"))
    else:
        print("  (no data)")

    # Table 2: S/C/N distribution
    print("\n" + "=" * 80)
    print("TABLE 2: S/C/N Distribution per Model")
    print("=" * 80)
    df2 = build_scn_table(all_results)
    print(df2.to_string(index=False))
    df2.to_csv(str(output_dir / "table2_scn_distribution.csv"), index=False)

    # Figure 1: Radar chart
    plot_radar_chart(all_results, str(output_dir / "fig1_radar_reliability.png"))

    # Figure 2: Evolution curve
    plot_evolution_curve(all_results, str(output_dir / "fig2_evolution_curve.png"))

    # Figure 3: Heatmap
    plot_scn_heatmap(all_results, str(output_dir / "fig3_contradiction_heatmap.png"))

    print(f"\nAll figures and tables saved to {output_dir}/")


if __name__ == "__main__":
    main()
