"""Figure 2 v2: Process Reward Pipeline — clean architecture diagram.
Same visual language as Fig 1: dashed teal borders, pastel sub-card tints.

Layout (no overlap, every label has a free landing zone):
  TOP band (3 columns):
    [Structured Response] ─►  [Process group (3 components)]   ┐
                              [Outcome group (2 components)]   ├─► [Σ] ─► [R(response)]
                              brackets/labels OUTSIDE cards
  BOTTOM band (4 columns):
    Four response qualities, each with two reward badges (process / binary).
    Callout below.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# palette
TEAL        = "#0F4C4C"
TEAL_MUTED  = "#4B7878"
GREEN       = "#22C55E"
GREEN_DARK  = "#16A34A"
RED         = "#DC2626"
WHITE       = "#FFFFFF"
GRAY_FILL   = "#F3F4F6"
MINT        = "#ECFDF5"
ICE         = "#EFF6FF"
CREAM       = "#FFFBEB"
PEACH       = "#FEF2F2"
LAVENDER    = "#F5F3FF"

W, H = 17, 9
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"

# ------------ helpers ------------
def dashed_box(ax, x, y, w, h, fill=GRAY_FILL, edgecolor=TEAL,
               lw=1.0, radius=0.12, dash=(4, 2)):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=lw, edgecolor=edgecolor, facecolor=fill,
                         linestyle=(0, dash))
    ax.add_patch(box)
    return box

def card(ax, x, y, w, h, title=None, body=None, fill=GRAY_FILL,
         emphasized=False, radius=0.12, title_size=11, body_size=8,
         title_color=TEAL, body_color=TEAL_MUTED, title_align="center"):
    lw = 1.6 if emphasized else 1.0
    dashed_box(ax, x, y, w, h, fill=fill, lw=lw, radius=radius)
    if title and body is None:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color)
    elif title and body:
        if title_align == "center":
            ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
                    fontsize=title_size, fontweight="bold", color=title_color)
            ax.text(x + w/2, y + 0.20, body, ha="center", va="bottom",
                    fontsize=body_size, color=body_color, style="italic")
        else:
            ax.text(x + 0.22, y + h - 0.18, title, ha="left", va="top",
                    fontsize=title_size, fontweight="bold", color=title_color)
            ax.text(x + 0.22, y + 0.13, body, ha="left", va="bottom",
                    fontsize=body_size, color=body_color)

def group(ax, x, y, w, h, label):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=0.2",
                         linewidth=1.8, edgecolor=TEAL, facecolor="none",
                         linestyle=(0, (5, 3)))
    ax.add_patch(box)
    ax.text(x + 0.55, y + h - 0.05, "  " + label + "  ",
            ha="left", va="center",
            fontsize=11.5, color=TEAL, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WHITE,
                      edgecolor="none"))

def sub_group(ax, x, y, w, h, label, color):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=0.14",
                         linewidth=1.5, edgecolor=color, facecolor="none",
                         linestyle=(0, (4, 2)))
    ax.add_patch(box)
    ax.text(x + 0.30, y + h - 0.05, "  " + label + "  ",
            ha="left", va="center",
            fontsize=10.5, color=color, fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE,
                      edgecolor="none"))

def arrow(ax, p1, p2, color=TEAL, lw=1.2, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

# ============================================================
# TOP BAND — Process Reward Pipeline
# ============================================================
group(ax, 0.4, 4.0, 16.2, 4.8, "1. Process Reward Pipeline")

# Structured Response input card
card(ax, 0.9, 5.4, 2.6, 2.4, "Structured\nResponse",
     body="evidence alignments\n+ reasoning chain\n+ label + diagnosis",
     fill=GRAY_FILL, emphasized=True, title_size=13, body_size=9)

# ===== Process sub-group (3 components) =====
proc_x, proc_y, proc_w = 4.5, 5.5, 4.6
proc_h = 2.6
sub_group(ax, proc_x, proc_y, proc_w, proc_h,
          "Process (70%)", GREEN_DARK)

comp_h = 0.62
comp_gap = 0.08
proc_components = [
    ("Rf  Format · w = 0.10",      "valid JSON · required fields present",   CREAM),
    ("Ra  Alignment · w = 0.30",   "per-span grounding · status validity",   MINT),
    ("Rc  Chain · w = 0.30",       "step judgments · evidence cited",        ICE),
]
proc_card_centers = []
proc_top = proc_y + proc_h - 0.45
for i, (title, body, fill) in enumerate(proc_components):
    cy = proc_top - (i + 1) * comp_h - i * comp_gap
    card(ax, proc_x + 0.3, cy, proc_w - 0.6, comp_h, title=title, body=body,
         fill=fill, radius=0.08, title_size=9.5, body_size=8,
         title_align="left")
    proc_card_centers.append(cy + comp_h / 2)

# ===== Outcome sub-group (2 components) =====
out_x_grp, out_y_grp, out_w_grp = 4.5, 4.05, 4.6
out_h_grp = 1.40
sub_group(ax, out_x_grp, out_y_grp, out_w_grp, out_h_grp,
          "Outcome (30%)", TEAL)

out_components = [
    ("Rl  Label · w = 0.15",       "predicted label matches gold",           LAVENDER),
    ("Rd  Diagnosis · w = 0.15",   "error type + actionable fix",            PEACH),
]
out_card_centers = []
out_top = out_y_grp + out_h_grp - 0.40
for i, (title, body, fill) in enumerate(out_components):
    cy = out_top - (i + 1) * comp_h - i * comp_gap * 0.6
    card(ax, out_x_grp + 0.3, cy, out_w_grp - 0.6, comp_h, title=title, body=body,
         fill=fill, radius=0.08, title_size=9.5, body_size=8,
         title_align="left")
    out_card_centers.append(cy + comp_h / 2)

# ===== Σ aggregator (consolidates all 5 components) =====
sum_x, sum_y = 10.2, 5.5
sum_w, sum_h = 1.4, 2.0
dashed_box(ax, sum_x, sum_y, sum_w, sum_h, fill=WHITE, lw=1.8)
ax.text(sum_x + sum_w / 2, sum_y + sum_h - 0.30, "Σ",
        ha="center", va="center", fontsize=32, color=TEAL,
        fontweight="bold")
ax.text(sum_x + sum_w / 2, sum_y + 0.32, "weighted\naggregate",
        ha="center", va="center", fontsize=9, color=TEAL_MUTED,
        style="italic", linespacing=1.2)

# Arrows: Response → process sub-group (just the group border, single arrow)
arrow(ax, (3.5, 6.6), (proc_x, proc_y + proc_h / 2), lw=1.3)

# Arrow: Response → outcome sub-group
arrow(ax, (3.5, 6.4), (out_x_grp, out_y_grp + out_h_grp / 2), lw=1.0, rad=-0.18)

# Arrows: process sub-group → Σ
arrow(ax, (proc_x + proc_w, proc_y + proc_h / 2),
          (sum_x, sum_y + sum_h * 0.65), lw=1.3, color=GREEN_DARK)

# Arrows: outcome sub-group → Σ
arrow(ax, (out_x_grp + out_w_grp, out_y_grp + out_h_grp / 2),
          (sum_x, sum_y + sum_h * 0.30), lw=1.3, color=TEAL)

# ===== R(response) output card =====
out_x, out_y = 12.4, 5.7
out_w, out_h = 3.7, 1.7
card(ax, out_x, out_y, out_w, out_h,
     title="R(response)",
     body="continuous reward ∈ [0, 1.28]",
     fill=MINT, emphasized=True,
     title_size=14, body_size=10)

# Σ → output arrow with formula label
arrow(ax, (sum_x + sum_w, sum_y + sum_h * 0.5),
          (out_x, out_y + out_h * 0.5), lw=1.6, color=GREEN)
ax.text((sum_x + sum_w + out_x) / 2, sum_y + sum_h * 0.5 + 0.35,
        "R = 0.10 Rf + 0.30 Ra + 0.30 Rc + 0.15 Rl + 0.15 Rd + Rcal",
        ha="center", va="center", fontsize=8.5, color=TEAL,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE,
                  edgecolor=GREEN, linewidth=0.8, linestyle="dashed"))

# ============================================================
# BOTTOM BAND — Reward Landscape on Four Response Qualities
# ============================================================
group(ax, 0.4, 0.3, 16.2, 3.4, "2. Reward Landscape Across Four Response Qualities")

qualities = [
    ("Q1", "correct label\n+ good reasoning",  1.13, 1.0,  TEAL),
    ("Q2", "good reasoning\nwrong label",      0.63, 0.0,  GREEN_DARK),
    ("Q3", "correct label\npoor reasoning",    0.28, 1.0,  TEAL),
    ("Q4", "unparseable\noutput",              0.00, 0.0,  TEAL),
]

q_w = 3.50
q_gap = (16.2 - q_w * 4) / 5
q_y_top, q_h = 1.65, 1.55
for i, (q, desc, proc_r, bin_r, accent) in enumerate(qualities):
    qx = 0.4 + q_gap + i * (q_w + q_gap)
    card(ax, qx, q_y_top, q_w, q_h,
         title=q,
         body=desc,
         fill=GRAY_FILL, title_size=13, body_size=9,
         emphasized=(q == "Q2"),
         title_color=accent)
    # Two reward badges below each quality card
    badge_y = q_y_top - 0.55
    is_q2 = (q == "Q2")
    ax.text(qx + q_w * 0.27, badge_y,
            f"process\nR ≈ {proc_r:.2f}",
            ha="center", va="center", fontsize=9.5,
            color=accent,
            fontweight="bold" if is_q2 else "normal",
            family="monospace", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.22",
                      facecolor=MINT if is_q2 else WHITE,
                      edgecolor=accent,
                      linewidth=1.5 if is_q2 else 0.7,
                      linestyle="solid" if is_q2 else "dashed"))
    ax.text(qx + q_w * 0.73, badge_y,
            f"binary\nR = {bin_r:.0f}",
            ha="center", va="center", fontsize=9.5,
            color=TEAL_MUTED, style="italic", family="monospace",
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=WHITE,
                      edgecolor=TEAL_MUTED, linewidth=0.7,
                      linestyle="dashed"))

# Callout: Q2 is the gradient-bearing case
ax.text(W / 2, 0.5,
        "★  Q2 (good reasoning, wrong label) is the only place where the GRPO gradient lives.  "
        "Process distinguishes  0.63  from  0.00  and  1.13 ;  binary collapses Q2 into the 0 spike  (Δ = 0).",
        ha="center", va="center", fontsize=10, color=GREEN_DARK,
        style="italic", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=WHITE,
                  edgecolor=GREEN_DARK, linewidth=1.2, linestyle="dashed"))

plt.savefig("fig2_reward_v2.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("fig2_reward_v2.png", bbox_inches="tight", pad_inches=0.15, dpi=180)
print("OK")
