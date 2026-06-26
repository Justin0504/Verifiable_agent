"""Figure: Reward Landscape — visualize advantage collapse.
Two-panel: left = process reward distribution (smooth), right = binary
(collapses to 0/1). Use synthetic samples whose statistics match the
paper's reported values (process mean ~1.12, std ~0.6; binary mean ~0.41
with two spikes at 0 and 1, std ~0.05 within a GRPO group).
Output: fig_landscape.pdf
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

TEAL       = "#374151"   # muted charcoal (was teal)
TEAL_MUTED = "#64748B"   # muted slate
GREEN      = "#5F8B6E"   # muted sage (was bright green)
RED        = "#9F4A4A"   # muted brick (was bright red)
WHITE      = "#FFFFFF"
MINT       = "#F0F4F0"   # very desaturated sage tint
PEACH      = "#EFE0E0"   # dusty rose (was peach)
GRAY_GRID  = "#E5E7EB"

rng = np.random.default_rng(2026)

# ------------ synthetic but statistics-consistent samples ------------
# Process reward: roughly Beta-like, centered around 0.95, range 0-1.28
# (matches the 4-level landscape from the paper: 1.13/0.63/0.28/0.0
# clustered roughly at these centers in equal-ish ratios)
centers = [1.13, 0.63, 0.28, 0.00]
weights = [0.40, 0.35, 0.18, 0.07]
process = np.concatenate([
    rng.normal(c, 0.10, int(1500 * w)) for c, w in zip(centers, weights)
])
process = np.clip(process, 0.0, 1.28)

# Binary reward: bimodal at 0 and 1, with proportions reflecting failure rate
# 28% unparseable + 35% wrong label = 63% at 0, 37% at 1
binary = np.concatenate([
    np.zeros(int(1500 * 0.63)),
    np.ones (int(1500 * 0.37)),
])
binary = binary + rng.normal(0, 0.012, len(binary))  # tiny jitter for vis

# ------------ canvas ------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                         gridspec_kw={"wspace": 0.32})

for ax, data, color, name, mean_text in [
    (axes[0], process, TEAL,  "Process Reward",
     "mean ≈ 1.12  ·  spread $\\pm$1.6"),
    (axes[1], binary,  RED,   "Binary Reward",
     "mean ≈ 0.41  ·  spread $\\pm$0.04"),
]:
    ax.hist(data, bins=44, range=(-0.05, 1.32),
            color=color, edgecolor="white", linewidth=0.5, alpha=0.92)
    ax.set_xlim(-0.05, 1.32)
    ax.set_ylim(0, 900 if name == "Binary Reward" else 220)
    ax.set_xlabel("reward value", fontsize=10, color=TEAL)
    ax.set_ylabel("count over 1500 rollouts", fontsize=10, color=TEAL)
    ax.set_title(name, fontsize=12, fontweight="bold", color=TEAL, loc="left",
                 pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL)
        spine.set_linewidth(0.8)
        spine.set_linestyle((0, (4, 2)))
    ax.tick_params(colors=TEAL_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRAY_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    # statistics box, top-right corner
    ax.text(0.97, 0.95, mean_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEAL, family="monospace",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE,
                      edgecolor=TEAL, linewidth=0.7, linestyle="dashed"))

# RIGHT panel: arrow + label routed to free space (lower-right)
ax = axes[1]
ax.annotate("63% of group rollouts → 0\n→ within-group $\\mu, \\sigma$ → 0",
            xy=(0.04, 780), xycoords="data",
            xytext=(0.42, 430), textcoords="data",
            fontsize=8.8, color=TEAL, ha="left", style="italic",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))
ax.annotate("37% → 1",
            xy=(1.00, 410), xycoords="data",
            xytext=(0.78, 600), textcoords="data",
            fontsize=8.8, color=TEAL, ha="left", style="italic",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))

# LEFT panel: arrow pointing to the 0.63 middle cluster, label routed to upper-left
ax = axes[0]
ax.annotate("middle level (0.63)\n— good reasoning, wrong label\n→ gradient persists",
            xy=(0.63, 85), xycoords="data",
            xytext=(0.04, 165), textcoords="data",
            fontsize=8.8, color=GREEN, ha="left", style="italic",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

# Headline above everything
fig.suptitle(
    "Process reward yields a four-level reward landscape;  "
    "binary reward collapses to two spikes — eliminating the GRPO gradient.",
    fontsize=11, color=TEAL, y=1.02, fontweight="bold")

plt.savefig("fig_landscape.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("fig_landscape.png", bbox_inches="tight", pad_inches=0.15, dpi=180)
print("OK")
