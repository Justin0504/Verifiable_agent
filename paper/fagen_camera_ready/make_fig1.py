"""Generate Fig 1 v3 (SEVA overview) — clean dashed style + concrete examples
+ pastel sub-card tints + Reflect weakness snapshot. No cartoons, no icons.
Output: fig1_overview_v2.pdf  (overwrites prior version, same filename so
LaTeX doesn't need to change).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- palette ----------
TEAL        = "#0F4C4C"
TEAL_MUTED  = "#4B7878"
GREEN       = "#22C55E"
WHITE       = "#FFFFFF"
GRAY_FILL   = "#F3F4F6"

# pastel tints (very light, infographic, non-Canva)
MINT     = "#ECFDF5"   # green tint  -> Evidence Alignment, Refine
ICE      = "#EFF6FF"   # blue tint   -> Reasoning Chain
CREAM    = "#FFFBEB"   # yellow tint -> Label+Conf, Reflect
PEACH    = "#FEF2F2"   # red tint    -> Error Diagnosis, Probe
LAVENDER = "#F5F3FF"   # purple tint -> Verify

# ---------- canvas ----------
W, H = 17, 9
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- helpers ----------
def card(ax, x, y, w, h, title=None, body=None, fill=GRAY_FILL,
         emphasized=False, radius=0.12, title_size=11, body_size=8,
         title_color=TEAL, body_color=TEAL_MUTED, title_align="center"):
    lw = 1.6 if emphasized else 1.0
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=lw, edgecolor=TEAL, facecolor=fill,
                         linestyle=(0, (4, 2)))
    ax.add_patch(box)
    if title and body is None:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color)
    elif title and body:
        if title_align == "center":
            ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
                    fontsize=title_size, fontweight="bold", color=title_color)
            ax.text(x + w/2, y + 0.18, body, ha="center", va="bottom",
                    fontsize=body_size, color=body_color, style="italic")
        else:  # left align — used for sub-cards with concrete examples
            ax.text(x + 0.22, y + h - 0.18, title, ha="left", va="top",
                    fontsize=title_size, fontweight="bold", color=title_color)
            ax.text(x + 0.22, y + 0.13, body, ha="left", va="bottom",
                    fontsize=body_size, color=body_color, family="DejaVu Sans Mono")

def group(ax, x, y, w, h, label, radius=0.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=1.8, edgecolor=TEAL, facecolor="none",
                         linestyle=(0, (5, 3)))
    ax.add_patch(box)
    ax.text(x + 0.55, y + h - 0.05, "  " + label + "  ",
            ha="left", va="center",
            fontsize=11.5, color=TEAL, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WHITE,
                      edgecolor="none"))

def arrow(ax, p1, p2, color=TEAL, lw=1.2, dashed=False, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14,
                        connectionstyle=f"arc3,rad={rad}",
                        linestyle="dashed" if dashed else "solid")
    ax.add_patch(a)

# ============================================================
# UPPER BAND
# ============================================================
group(ax, 0.4, 4.3, 16.2, 4.5, "1. Structured Verification")

# Column A — inputs with concrete claim/source
card(ax, 0.9, 6.7, 2.7, 1.7, "Claim",
     body='"60% of participants\nsignificantly improved"',
     fill=ICE, title_size=12.5, body_size=9)
card(ax, 0.9, 4.8, 2.7, 1.7, "Source",
     body='"60% of subjects\nshowed improvement"',
     fill=MINT, title_size=12.5, body_size=9)

# Column B — SEVA Verifier (text only, no human/icon)
card(ax, 5.1, 5.8, 3.1, 1.7, "SEVA Verifier",
     body="Qwen2.5  ·  Process-Reward GRPO",
     fill=WHITE, emphasized=True, title_size=15.5, body_size=10)

# Column C — Structured Output container
out_x, out_y, out_w, out_h = 9.7, 4.5, 6.6, 4.1
container = FancyBboxPatch((out_x, out_y), out_w, out_h,
                            boxstyle="round,pad=0,rounding_size=0.18",
                            linewidth=1.4, edgecolor=TEAL,
                            facecolor=WHITE, linestyle=(0, (4, 2)))
ax.add_patch(container)
ax.text(out_x + 0.35, out_y + out_h - 0.05, "  Structured Output  ",
        ha="left", va="center", fontsize=11.5, fontweight="bold", color=TEAL,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE, edgecolor="none"))

# 4 sub-cards with concrete content + pastel tints
sub_h = 0.78
sub_specs = [
    ("Evidence Alignment", '✓ "60% participants" → "60% subjects" [match]\n✗ "significantly improved" → NOT_FOUND',  MINT),
    ("Reasoning Chain",    'Step 1: "60% matches" → supported\nStep 2: "significantly" has no source → ✗',           ICE),
    ("Label + Confidence", "Not Attributable      confidence γ = 0.85",                                              CREAM),
    ("Error Diagnosis",    'type: scope_inflation     fix: remove "significantly"',                                  PEACH),
]
sub_top = out_y + out_h - 0.60
for i, (t, b, fc) in enumerate(sub_specs):
    sy = sub_top - (i + 1) * sub_h - i * 0.09
    sub = FancyBboxPatch((out_x + 0.3, sy), out_w - 0.6, sub_h,
                          boxstyle="round,pad=0,rounding_size=0.08",
                          linewidth=0.9, edgecolor=TEAL,
                          facecolor=fc, linestyle=(0, (4, 2)))
    ax.add_patch(sub)
    ax.text(out_x + 0.5, sy + sub_h - 0.18, t, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color=TEAL)
    ax.text(out_x + 0.5, sy + 0.13, b, ha="left", va="bottom",
            fontsize=8.3, color=TEAL_MUTED, family="DejaVu Sans Mono",
            linespacing=1.35)
    if i == 3:
        err_diag_y_mid = sy + sub_h * 0.5
        err_diag_x_left = out_x + 0.3

# Upper-band arrows
arrow(ax, (3.6, 7.45), (5.1, 6.95))   # Claim → Verifier
arrow(ax, (3.6, 5.6),  (5.1, 6.4))    # Source → Verifier
arrow(ax, (8.2, 6.6),  (out_x, 6.2))  # Verifier → Output (avoid title)

# ============================================================
# LOWER BAND
# ============================================================
group(ax, 0.4, 0.3, 16.2, 3.7, "2. Self-Evolution Loop")

nodes = [
    ("Verify",  "run model on held-out\nclaims; collect structured\npredictions",                LAVENDER),
    ("Reflect", "build 6-bin weakness\nprofile from per-claim\nerror diagnoses",                  CREAM),
    ("Probe",   "generate adversarial\nsamples ∝ per-category\nweakness budget",                  PEACH),
    ("Refine",  "fine-tune verifier with\nGRPO + process reward;\nmerge with replay set",         MINT),
]
node_w, node_h = 3.3, 2.0
node_y = 1.45
n = len(nodes)
gap = (16.2 - node_w * n) / (n + 1)
node_xs = []
for i, (t, b, fc) in enumerate(nodes):
    x = 0.4 + gap + i * (node_w + gap)
    node_xs.append(x)
    card(ax, x, node_y, node_w, node_h, t, body=b,
         fill=fc, emphasized=True, title_size=14, body_size=9.2)

# Forward arrows between loop nodes
for i in range(n - 1):
    arrow(ax, (node_xs[i] + node_w, node_y + node_h/2),
              (node_xs[i+1],         node_y + node_h/2))

# Loop-back: Refine -> Verify, curve under the band
p1 = (node_xs[-1] + node_w/2, node_y)
p2 = (node_xs[0]  + node_w/2, node_y)
loopback = FancyArrowPatch(p1, p2, arrowstyle="-|>", color=TEAL, lw=1.2,
                            mutation_scale=14,
                            connectionstyle="arc3,rad=0.30")
ax.add_patch(loopback)
ax.text((p1[0] + p2[0]) / 2, 0.7, "iterate",
        ha="center", va="center", fontsize=9.5,
        color=TEAL_MUTED, style="italic")

# ============================================================
# CROSS-BAND green feedback: Error Diagnosis → Reflect
# Route: exit from bottom of Error Diagnosis card, curve DOWN through
# the band-gap (below Structured Output, above the lower band group
# border), then UP into Reflect's top edge.
# ============================================================
err_diag_x_center = out_x + out_w * 0.5
src_x = err_diag_x_center
src_y = err_diag_y_mid - 0.30   # start just below the Error Diagnosis card
dst_x = node_xs[1] + node_w * 0.5
dst_y = node_y + node_h         # top of Reflect node
green_arrow = FancyArrowPatch(
    (src_x, src_y), (dst_x, dst_y),
    arrowstyle="-|>", color=GREEN, lw=1.8, mutation_scale=16,
    connectionstyle="arc3,rad=-0.30",   # arc bulges DOWN through gap
    linestyle=(0, (4, 3))
)
ax.add_patch(green_arrow)
ax.text(8.6, 4.05, "  structured errors  →  guide refinement  ",
        ha="center", va="center", fontsize=10, color=GREEN,
        style="italic", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                  edgecolor=GREEN, linewidth=0.8))

# Save
plt.savefig("fig1_overview_v2.pdf", bbox_inches="tight", pad_inches=0.1)
plt.savefig("fig1_overview_v2.png", bbox_inches="tight", pad_inches=0.1, dpi=200)
print("OK")
