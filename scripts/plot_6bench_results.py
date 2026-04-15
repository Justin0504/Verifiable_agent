"""Generate 6-benchmark comparison figures for paper.

Reads results from grpo/sft/zeroshot_6bench_eval and produces:
1. Grouped bar chart (6 benchmarks × 3 models)
2. Radar chart
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Load results
# ============================================================
base = "/Users/justin/Verifiable_agent/results"

models = {}
for name, dname in [("Zero-shot", "zeroshot_6bench_eval"), ("SFT", "sft_6bench_eval"), ("GRPO", "grpo_6bench_eval")]:
    with open(f"{base}/{dname}/summary.json") as f:
        data = json.load(f)
    models[name] = {k: v["accuracy"] * 100 for k, v in data["results"].items()}

benchmarks = ["fever", "truthfulqa", "scifact", "halueval", "musique", "factscore"]
bench_labels = ["FEVER", "TruthfulQA", "SciFact", "HaluEval", "MuSiQue", "FActScore"]

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    "Zero-shot": "#90CAF9",
    "SFT": "#FFB74D",
    "GRPO": "#4CAF50",
}

# ============================================================
# Figure 1: Grouped Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(benchmarks))
width = 0.25
model_names = ["Zero-shot", "SFT", "GRPO"]

for i, model in enumerate(model_names):
    vals = [models[model].get(b, 0) for b in benchmarks]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=model, color=COLORS[model], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Fact Verification Accuracy Across 6 Benchmarks', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(bench_labels, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# Add average line annotations
for model in model_names:
    vals = [models[model].get(b, 0) for b in benchmarks]
    avg = sum(vals) / len(vals)
    # Small text at bottom

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_comparison.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_comparison.png')
print("Saved: 6bench_comparison.pdf/png")


# ============================================================
# Figure 2: Radar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Radar setup
angles = np.linspace(0, 2 * np.pi, len(benchmarks), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

for model in model_names:
    vals = [models[model].get(b, 0) for b in benchmarks]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=COLORS[model], markersize=6)
    ax.fill(angles, vals, alpha=0.1, color=COLORS[model])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(bench_labels, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
ax.set_title('Benchmark Performance Radar', fontweight='bold', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_radar.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_radar.png')
print("Saved: 6bench_radar.pdf/png")


# ============================================================
# Figure 3: Delta improvement chart (GRPO vs SFT, GRPO vs Zero-shot)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

grpo_vals = [models["GRPO"].get(b, 0) for b in benchmarks]
sft_vals = [models["SFT"].get(b, 0) for b in benchmarks]
zs_vals = [models["Zero-shot"].get(b, 0) for b in benchmarks]

delta_vs_sft = [g - s for g, s in zip(grpo_vals, sft_vals)]
delta_vs_zs = [g - z for g, z in zip(grpo_vals, zs_vals)]

x = np.arange(len(benchmarks))
width = 0.35

bars1 = ax.bar(x - width/2, delta_vs_zs, width, label='GRPO vs Zero-shot', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, delta_vs_sft, width, label='GRPO vs SFT', color='#FF9800', alpha=0.85)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 2 if h >= 0 else -2
        ax.annotate(f'{h:+.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Accuracy Difference (%)')
ax.set_title('GRPO Improvement Over Baselines', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(bench_labels, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_delta.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/6bench_delta.png')
print("Saved: 6bench_delta.pdf/png")


# ============================================================
# Print summary table (for paper)
# ============================================================
print("\n" + "=" * 75)
print("TABLE 1: Accuracy (%) across 6 benchmarks")
print("=" * 75)
header = f"{'Model':<12}" + "".join(f"{b:>12}" for b in bench_labels) + f"{'Avg':>8}"
print(header)
print("-" * 75)
for model in model_names:
    vals = [models[model].get(b, 0) for b in benchmarks]
    avg = sum(vals) / len(vals)
    row = f"{model:<12}" + "".join(f"{v:>12.1f}" for v in vals) + f"{avg:>8.1f}"
    print(row)
print("=" * 75)
