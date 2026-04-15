"""Generate paper-quality plots for GRPO training analysis.

Reads raw training log metrics and produces:
1. Policy loss curve
2. Reward (score) curve
3. Entropy curve
4. Gradient norm curve
5. Validation reward curve
6. Combined accuracy comparison (zero-shot vs SFT vs GRPO)
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Parse training metrics from raw log lines
# ============================================================
RAW_LOG = "/tmp/grpo_metrics_raw.txt"

steps, pg_losses, entropies, grad_norms = [], [], [], []
score_means, score_maxs, score_mins = [], [], []
epochs, lrs = [], []
response_lens = []

with open(RAW_LOG) as f:
    for line in f:
        line = line.strip()
        # Extract key metrics using regex
        def extract(key):
            m = re.search(rf'{key}:([-\d.eE+]+)', line)
            return float(m.group(1)) if m else None

        step = extract('training/global_step')
        if step is None:
            continue

        steps.append(int(step))
        pg_losses.append(extract('actor/pg_loss'))
        entropies.append(extract('actor/entropy'))
        grad_norms.append(extract('actor/grad_norm'))
        score_means.append(extract('critic/score/mean'))
        score_maxs.append(extract('critic/score/max'))
        score_mins.append(extract('critic/score/min'))
        epochs.append(extract('training/epoch'))
        lrs.append(extract('actor/lr'))
        response_lens.append(extract('response_length/mean'))

steps = np.array(steps)
pg_losses = np.array(pg_losses)
entropies = np.array(entropies)
grad_norms = np.array(grad_norms)
score_means = np.array(score_means)
response_lens = np.array(response_lens)

# Validation metrics (at step 0, 25, 50, 75, 100, 105)
val_steps = [0, 25, 50, 75, 100, 105]
val_rewards = [1.2906, 1.3377, 1.3835, 1.3948, 1.3732, 1.3984]

# ============================================================
# Plot styling
# ============================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'blue': '#2196F3',
    'red': '#F44336',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'purple': '#9C27B0',
    'teal': '#009688',
}

# Epoch boundaries
epoch_boundaries = [35, 70]  # 105 steps / 3 epochs = 35 steps/epoch

def add_epoch_lines(ax):
    for eb in epoch_boundaries:
        ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    # Add epoch labels
    ax.text(17, ax.get_ylim()[1], 'Epoch 0', ha='center', va='top', fontsize=8, color='gray')
    ax.text(52, ax.get_ylim()[1], 'Epoch 1', ha='center', va='top', fontsize=8, color='gray')
    ax.text(87, ax.get_ylim()[1], 'Epoch 2', ha='center', va='top', fontsize=8, color='gray')

# ============================================================
# Figure 1: Training Dynamics (2x2 subplot)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('GRPO Training Dynamics — Qwen2.5-3B Fact Verification', fontsize=15, fontweight='bold')

# 1a: Policy Loss
ax = axes[0, 0]
ax.plot(steps, pg_losses, color=COLORS['blue'], linewidth=1, alpha=0.6)
# Smoothed
window = 5
smoothed = np.convolve(pg_losses, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], smoothed, color=COLORS['blue'], linewidth=2, label='Smoothed (w=5)')
ax.set_xlabel('Training Step')
ax.set_ylabel('Policy Gradient Loss')
ax.set_title('(a) Policy Loss')
for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.legend()

# 1b: Reward Score
ax = axes[0, 1]
ax.plot(steps, score_means, color=COLORS['green'], linewidth=1, alpha=0.5)
smoothed_score = np.convolve(score_means, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], smoothed_score, color=COLORS['green'], linewidth=2, label='Train Reward (smoothed)')
ax.plot(val_steps, val_rewards, 'D-', color=COLORS['red'], markersize=6, linewidth=2, label='Val Reward')
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Reward Score')
ax.set_title('(b) Reward Score')
for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.legend()

# 1c: Entropy
ax = axes[1, 0]
ax.plot(steps, entropies, color=COLORS['purple'], linewidth=1, alpha=0.5)
smoothed_ent = np.convolve(entropies, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], smoothed_ent, color=COLORS['purple'], linewidth=2, label='Smoothed')
ax.set_xlabel('Training Step')
ax.set_ylabel('Policy Entropy')
ax.set_title('(c) Policy Entropy')
for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.legend()

# 1d: Gradient Norm
ax = axes[1, 1]
ax.plot(steps, grad_norms, color=COLORS['orange'], linewidth=1, alpha=0.5)
smoothed_gn = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], smoothed_gn, color=COLORS['orange'], linewidth=2, label='Smoothed')
ax.set_xlabel('Training Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('(d) Gradient Norm')
for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.legend()

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/grpo_training_dynamics.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/grpo_training_dynamics.png')
print("Saved: grpo_training_dynamics.pdf/png")

# ============================================================
# Figure 2: Accuracy Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Zero-shot\n(Qwen2.5-3B)', 'SFT\n(GPT-4o distill)', 'GRPO\n(RL trained)']
# Actual evaluation results from test set (559 samples)
overall_acc = [62.8, 76.6, 84.1]
s_acc = [62.6, 74.3, 92.0]
c_acc = [40.1, 92.2, 81.2]
n_acc = [87.2, 62.2, 78.9]

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, overall_acc, width, label='Overall', color=COLORS['blue'], alpha=0.85)
bars2 = ax.bar(x - 0.5*width, s_acc, width, label='S (Supported)', color=COLORS['green'], alpha=0.85)
bars3 = ax.bar(x + 0.5*width, c_acc, width, label='C (Contradicted)', color=COLORS['orange'], alpha=0.85)
bars4 = ax.bar(x + 1.5*width, n_acc, width, label='N (Not Enough Info)', color=COLORS['red'], alpha=0.85)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# GPT-4o-mini reference line
ax.axhline(y=85.6, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(2.5, 86.2, 'GPT-4o-mini (85.6%)', ha='right', va='bottom', fontsize=9, color='gray', style='italic')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Fact Verification Accuracy: Zero-shot → SFT → GRPO', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/accuracy_comparison.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/accuracy_comparison.png')
print("Saved: accuracy_comparison.pdf/png")

# ============================================================
# Figure 3: Validation Reward Across Training
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(val_steps, val_rewards, 'o-', color=COLORS['blue'], linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Validation Reward')

# Annotate key points
ax.annotate(f'Initial: {val_rewards[0]:.3f}', (val_steps[0], val_rewards[0]),
            xytext=(15, -20), textcoords='offset points', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'Best: {max(val_rewards):.3f}',
            (val_steps[val_rewards.index(max(val_rewards))], max(val_rewards)),
            xytext=(15, 10), textcoords='offset points', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'Final: {val_rewards[-1]:.3f}', (val_steps[-1], val_rewards[-1]),
            xytext=(-60, -20), textcoords='offset points', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))

for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_xlabel('Training Step')
ax.set_ylabel('Validation Reward (mean)')
ax.set_title('Validation Reward During GRPO Training', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/validation_reward_curve.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/validation_reward_curve.png')
print("Saved: validation_reward_curve.pdf/png")

# ============================================================
# Figure 4: Response Length Distribution Over Training
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, response_lens, color=COLORS['teal'], linewidth=1, alpha=0.5)
smoothed_rl = np.convolve(response_lens, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], smoothed_rl, color=COLORS['teal'], linewidth=2, label='Mean Response Length')
ax.set_xlabel('Training Step')
ax.set_ylabel('Tokens')
ax.set_title('Mean Response Length During Training', fontweight='bold')
for eb in epoch_boundaries:
    ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.legend()

plt.tight_layout()
plt.savefig('/Users/justin/Verifiable_agent/figures/response_length.pdf')
plt.savefig('/Users/justin/Verifiable_agent/figures/response_length.png')
print("Saved: response_length.pdf/png")

print("\nAll figures saved to /Users/justin/Verifiable_agent/figures/")
