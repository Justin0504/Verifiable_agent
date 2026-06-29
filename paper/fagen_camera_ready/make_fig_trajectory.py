"""Figure: Per-Round Per-Benchmark F1 Trajectory.
Shows the monotone divergence: HaluEval rises, TruthfulQA falls,
ClearFacts/FEVER stay flat across Rounds 0-4.
Output: fig_trajectory.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

TEAL       = "#0F4C4C"
TEAL_MUTED = "#4B7878"
WHITE      = "#FFFFFF"
GRAY_GRID  = "#E5E7EB"

# Colors echoing fig1 pastels but saturated enough for line plot
COL_CF    = "#374151"   # muted charcoal (was deep teal)
COL_FEVER = "#5B7A92"   # muted slate-blue (was bright blue)
COL_TQA   = "#9F4A4A"   # muted brick (was bright red)
COL_HE    = "#5F8B6E"   # muted sage (was bright green)

rounds = ["R0 (Step150)", "R1 (rules)", "R2 (LoRA, 1.1K)",
          "R3 (Full FT, 2.0K)", "R4 (mega-FT, 7.8K)"]
x = np.arange(5)

# Absolute F1 numbers (from ARR paper, line 314 + tab:specialization deltas)
cf    = [65.2, 64.5, 66.5, 65.2, 65.1]
fever = [90.7, 90.2, 92.3, 91.9, 92.2]
tqa   = [68.8, 69.9, 58.6, 55.0, 56.4]
he    = [57.1, 57.7, 68.0, 71.4, 72.0]

fig, ax = plt.subplots(figsize=(9.6, 5.0))

def plot_line(ax, x, y, color, label, marker):
    ax.plot(x, y, color=color, lw=2.0, marker=marker, markersize=8,
            markerfacecolor=WHITE, markeredgewidth=1.8,
            markeredgecolor=color, label=label, zorder=3)

plot_line(ax, x, fever, COL_FEVER, "FEVER",      "s")
plot_line(ax, x, he,    COL_HE,    "HaluEval",   "^")
plot_line(ax, x, tqa,   COL_TQA,   "TruthfulQA", "v")
plot_line(ax, x, cf,    COL_CF,    "ClearFacts", "o")

# axes
ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=9.5, color=TEAL)
ax.set_ylabel("macro F1", fontsize=11, color=TEAL)
ax.set_xlabel("Self-Evolution round", fontsize=11, color=TEAL, labelpad=8)
ax.set_ylim(50, 95)
ax.set_xlim(-0.3, 4.3)
ax.grid(axis="y", color=GRAY_GRID, linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
ax.tick_params(colors=TEAL_MUTED, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(TEAL)
    spine.set_linewidth(0.9)
    spine.set_linestyle((0, (4, 2)))

# legend outside, on the right
leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=10, frameon=True,
                handlelength=2.2, borderpad=0.8, labelspacing=0.8)
leg.get_frame().set_edgecolor(TEAL)
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_linestyle("dashed")
leg.get_frame().set_facecolor(WHITE)
for txt in leg.get_texts():
    txt.set_color(TEAL)

# Annotations: highlight the asymmetric divergence on R2-R4
ax.annotate("HaluEval: monotone $+10.9 \\to +14.9$ pp",
            xy=(3.85, 71.7), xytext=(1.7, 81.5),
            fontsize=9.5, color=COL_HE, ha="left", style="italic",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL_HE, lw=1.0))

ax.annotate("TruthfulQA: monotone $-10.2 \\to -12.4$ pp",
            xy=(3.85, 56.4), xytext=(1.7, 51.5),
            fontsize=9.5, color=COL_TQA, ha="left", style="italic",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL_TQA, lw=1.0))

# shaded band emphasizing the "divergence zone" R2-R4
ax.axvspan(1.5, 4.3, color=GRAY_GRID, alpha=0.35, zorder=1)
ax.text(3.9, 94.0, "specialization regime",
        ha="right", va="top", fontsize=9, color=TEAL_MUTED,
        style="italic")

# title
fig.suptitle(
    "Self-evolution drives a signed, monotone specialization — "
    "not a uniform improvement.",
    fontsize=11.5, color=TEAL, y=0.98, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig_trajectory.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("fig_trajectory.png", bbox_inches="tight", pad_inches=0.15, dpi=180)
print("OK")
