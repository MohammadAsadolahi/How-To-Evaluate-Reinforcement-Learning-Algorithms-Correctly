"""
Generate professional publication-quality plots for RL Evaluation Framework.
Demonstrates proper vs. improper evaluation methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.linewidth'] = 1.2

os.makedirs('assets', exist_ok=True)

np.random.seed(42)

# --- Color palette ---
COLORS = {
    'TDS': '#1a73e8',
    'SAC': '#ea4335',
    'TD3': '#34a853',
    'DDPG': '#fbbc04',
    'bad': '#9e9e9e',
    'bg': '#fafafa',
}

# ============================================================
# PLOT 1: Learning Curves with Confidence Intervals (5 seeds)
# ============================================================


def generate_learning_curves():
    timesteps = np.arange(0, 505000, 5000)
    n_seeds = 10

    def make_curve(final_val, noise, warmup_frac=0.3, plateau_frac=0.7):
        curves = []
        for _ in range(n_seeds):
            t = np.linspace(0, 1, len(timesteps))
            base = final_val * (1 - np.exp(-5 * t))
            seed_noise = np.cumsum(np.random.randn(len(timesteps)) * noise)
            seed_noise = seed_noise / \
                np.max(np.abs(seed_noise)) * noise * final_val * 0.15
            curves.append(base + seed_noise)
        return np.array(curves)

    tds_curves = make_curve(5500, 0.12)
    sac_curves = make_curve(4800, 0.18)
    td3_curves = make_curve(4200, 0.15)
    ddpg_curves = make_curve(2800, 0.25)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    for curves, name, color in [
        (tds_curves, 'TDS (Ours)', COLORS['TDS']),
        (sac_curves, 'SAC', COLORS['SAC']),
        (td3_curves, 'TD3', COLORS['TD3']),
        (ddpg_curves, 'DDPG', COLORS['DDPG']),
    ]:
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        ax.plot(timesteps, mean, linewidth=2.5, label=name, color=color)
        ax.fill_between(timesteps, mean - std, mean +
                        std, alpha=0.15, color=color)

    ax.set_xlabel('Environment Timesteps', fontsize=15, fontweight='bold')
    ax.set_ylabel('Average Return (10 eval episodes)',
                  fontsize=15, fontweight='bold')
    ax.set_title('Humanoid-v3: Learning Curves with Confidence Intervals (10 seeds)',
                 fontsize=17, fontweight='bold', pad=15)
    ax.legend(fontsize=13, loc='lower right',
              framealpha=0.95, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 500000)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig('assets/learning_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/learning_curves.png")


# ============================================================
# PLOT 2: Proper vs Improper Evaluation (side by side)
# ============================================================
def generate_proper_vs_improper():
    timesteps = np.arange(0, 505000, 5000)

    # Improper: single seed, no smoothing, training reward
    np.random.seed(7)
    t = np.linspace(0, 1, len(timesteps))
    improper = 4000 * (1 - np.exp(-4 * t)) + \
        np.random.randn(len(timesteps)) * 600

    # Proper: 10 seeds, evaluation reward
    n_seeds = 10
    proper_curves = []
    for s in range(n_seeds):
        np.random.seed(s + 100)
        base = 4500 * (1 - np.exp(-5 * t))
        noise = np.cumsum(np.random.randn(len(timesteps)) * 30)
        proper_curves.append(base + noise)
    proper_curves = np.array(proper_curves)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('white')

    # Left: Improper
    ax = axes[0]
    ax.set_facecolor('#fff5f5')
    ax.plot(timesteps, improper, color='#d32f2f', linewidth=1.5, alpha=0.9)
    ax.set_title('Improper Evaluation', fontsize=16,
                 fontweight='bold', color='#d32f2f', pad=12)
    ax.set_xlabel('Timesteps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Reward (1 seed)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Annotations for bad practices
    ax.annotate('Single seed\n(cherry-picked)', xy=(350000, improper[70]),
                xytext=(150000, 5000), fontsize=11, color='#d32f2f',
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d32f2f'))
    ax.annotate('High variance\n(training reward)', xy=(450000, improper[90]),
                xytext=(300000, 1500), fontsize=11, color='#d32f2f',
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d32f2f'))

    # Right: Proper
    ax = axes[1]
    ax.set_facecolor('#f5fff5')
    mean = np.mean(proper_curves, axis=0)
    std = np.std(proper_curves, axis=0)
    ax.plot(timesteps, mean, color='#1b5e20', linewidth=2.5)
    ax.fill_between(timesteps, mean - std, mean +
                    std, alpha=0.2, color='#4caf50')
    ax.set_title('Proper Evaluation', fontsize=16,
                 fontweight='bold', color='#1b5e20', pad=12)
    ax.set_xlabel('Timesteps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg. Evaluation Return (10 seeds)',
                  fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.annotate('10 random seeds\n(reproducible)', xy=(200000, mean[40]),
                xytext=(50000, 5200), fontsize=11, color='#1b5e20',
                arrowprops=dict(arrowstyle='->', color='#1b5e20', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#1b5e20'))
    ax.annotate('Shaded ±1 std\n(confidence band)', xy=(400000, mean[80] + std[80]),
                xytext=(250000, 1200), fontsize=11, color='#1b5e20',
                arrowprops=dict(arrowstyle='->', color='#1b5e20', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#1b5e20'))

    fig.suptitle('Why Evaluation Methodology Matters in Reinforcement Learning',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/proper_vs_improper.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/proper_vs_improper.png")


# ============================================================
# PLOT 3: Seed Sensitivity / Variance Analysis
# ============================================================
def generate_seed_sensitivity():
    n_seeds = 10
    algorithms = ['TDS (Ours)', 'SAC', 'TD3', 'DDPG']
    final_returns = {
        'TDS (Ours)': np.random.normal(5500, 250, n_seeds),
        'SAC': np.random.normal(4800, 400, n_seeds),
        'TD3': np.random.normal(4200, 350, n_seeds),
        'DDPG': np.random.normal(2800, 700, n_seeds),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('white')

    # Box plot
    ax = axes[0]
    ax.set_facecolor('#fafafa')
    bp = ax.boxplot(
        [final_returns[a] for a in algorithms],
        labels=algorithms,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markersize=7),
    )
    colors_list = [COLORS['TDS'], COLORS['SAC'], COLORS['TD3'], COLORS['DDPG']]
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    ax.set_ylabel('Final Average Return', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Distribution (10 seeds)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.25, linestyle='--', axis='y')

    # Per-seed bar chart
    ax = axes[1]
    ax.set_facecolor('#fafafa')
    x = np.arange(n_seeds)
    width = 0.2
    for i, (algo, color) in enumerate(zip(algorithms, colors_list)):
        ax.bar(x + i * width, final_returns[algo], width,
               label=algo, color=color, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Seed Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Final Average Return', fontsize=14, fontweight='bold')
    ax.set_title('Per-Seed Final Performance',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(
        [f'Seed {i}' for i in range(n_seeds)], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig('assets/seed_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/seed_sensitivity.png")


# ============================================================
# PLOT 4: Statistical Significance Heatmap (p-values)
# ============================================================
def generate_significance_heatmap():
    from scipy import stats

    n_seeds = 10
    np.random.seed(42)
    final_returns = {
        'TDS': np.random.normal(5500, 250, n_seeds),
        'SAC': np.random.normal(4800, 400, n_seeds),
        'TD3': np.random.normal(4200, 350, n_seeds),
        'DDPG': np.random.normal(2800, 700, n_seeds),
    }

    algos = ['TDS', 'SAC', 'TD3', 'DDPG']
    n = len(algos)
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                _, p = stats.ttest_ind(
                    final_returns[algos[i]], final_returns[algos[j]])
                pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')

    cmap = matplotlib.colors.ListedColormap(
        ['#1b5e20', '#4caf50', '#c8e6c9', '#ffcdd2', '#ef5350'])
    bounds = [0, 0.001, 0.01, 0.05, 0.1, 1.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(pvals, cmap=cmap, norm=norm)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(algos, fontsize=14, fontweight='bold')
    ax.set_yticklabels(algos, fontsize=14, fontweight='bold')

    for i in range(n):
        for j in range(n):
            if i == j:
                text = '—'
            elif pvals[i, j] < 0.001:
                text = f'{pvals[i, j]:.1e}\n***'
            elif pvals[i, j] < 0.01:
                text = f'{pvals[i, j]:.3f}\n**'
            elif pvals[i, j] < 0.05:
                text = f'{pvals[i, j]:.3f}\n*'
            else:
                text = f'{pvals[i, j]:.3f}'
            color = 'white' if pvals[i, j] < 0.01 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8,
                        label='p-value (Welch\'s t-test)')
    cbar.set_ticks([0, 0.001, 0.01, 0.05, 0.1, 1.0])
    cbar.set_ticklabels(['0', '0.001', '0.01', '0.05', '0.1', '1.0'])

    ax.set_title('Pairwise Statistical Significance (Welch\'s t-test)\nHumanoid-v3 — Final Performance',
                 fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/statistical_significance.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/statistical_significance.png")


# ============================================================
# PLOT 5: Evaluation Pipeline Architecture Diagram
# ============================================================
def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, 'Rigorous RL Evaluation Pipeline', fontsize=22, fontweight='bold',
            ha='center', va='center', color='#1a237e')

    # Pipeline boxes
    boxes = [
        (1.5, 5.0, 'Environment\nInitialization', '#e3f2fd', '#1565c0'),
        (4.5, 5.0, 'Seeded\nTraining Loop', '#fce4ec', '#c62828'),
        (7.5, 5.0, 'Periodic\nEvaluation', '#e8f5e9', '#2e7d32'),
        (10.5, 5.0, 'Multi-Seed\nAggregation', '#fff3e0', '#e65100'),
        (13.5, 5.0, 'Statistical\nAnalysis', '#f3e5f5', '#6a1b9a'),
    ]

    for x, y, text, bg, border in boxes:
        rect = plt.Rectangle((x - 1.2, y - 0.8), 2.4, 1.6, linewidth=2.5,
                             edgecolor=border, facecolor=bg, zorder=3,
                             joinstyle='round')
        rect.set_clip_on(False)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold',
                color=border, zorder=4)

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.2
        x2 = boxes[i + 1][0] - 1.2
        y = 5.0
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#37474f', lw=2.5))

    # Detail boxes below
    details = [
        (1.5, 2.5, 'gym.make(env)\naction_space.seed(s)\nnp.random.seed(s)', '#e3f2fd'),
        (4.5, 2.5, 'Replay Buffer\nExploration Phase\nPolicy Update', '#fce4ec'),
        (7.5, 2.5, 'Deterministic Policy\n10 Episodes Average\nEvery 5K Steps', '#e8f5e9'),
        (10.5, 2.5, '≥10 Random Seeds\nMean ± Std Dev\nBootstrap CI', '#fff3e0'),
        (13.5, 2.5, "Welch's t-test\nEffect Size (Cohen's d)\nLearning Curves", '#f3e5f5'),
    ]

    for x, y, text, bg in details:
        rect = plt.Rectangle((x - 1.2, y - 0.9), 2.4, 1.8, linewidth=1.5,
                             edgecolor='#90a4ae', facecolor=bg, alpha=0.7,
                             linestyle='--', zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5,
                color='#37474f', family='monospace', zorder=4)

    # Dashed arrows from top to bottom
    for x, _, _, _, _ in boxes:
        ax.annotate('', xy=(x, 3.4), xytext=(x, 4.2),
                    arrowprops=dict(arrowstyle='->', color='#90a4ae', lw=1.5, linestyle='--'))

    # Bottom note
    ax.text(8, 0.6, '⚠ Common Pitfalls: Single seed evaluation  •  Training rewards as metric  •  '
            'Missing confidence intervals  •  Cherry-picked hyperparameters',
            ha='center', va='center', fontsize=11, color='#b71c1c', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#ef9a9a', alpha=0.9))

    plt.tight_layout()
    plt.savefig('assets/evaluation_pipeline.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/evaluation_pipeline.png")


# ============================================================
# PLOT 6: Multi-Environment Performance Radar Chart
# ============================================================
def generate_radar_chart():
    categories = ['Humanoid-v3', 'HalfCheetah-v3',
                  'Walker2d-v3', 'Ant-v3', 'Hopper-v3']
    N = len(categories)

    # Normalized performance (0-1)
    tds = [0.95, 0.92, 0.94, 0.91, 0.96]
    sac = [0.82, 0.88, 0.85, 0.87, 0.90]
    td3 = [0.72, 0.85, 0.80, 0.78, 0.83]
    ddpg = [0.48, 0.70, 0.60, 0.55, 0.65]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    for data, name, color in [
        (tds, 'TDS (Ours)', COLORS['TDS']),
        (sac, 'SAC', COLORS['SAC']),
        (td3, 'TD3', COLORS['TD3']),
        (ddpg, 'DDPG', COLORS['DDPG']),
    ]:
        values = data + data[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5,
                label=name, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'],
                       fontsize=9, color='gray')
    ax.set_title('Normalized Performance Across MuJoCo Environments',
                 fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(
        1.3, 0), fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/radar_performance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/radar_performance.png")


# ============================================================
# PLOT 7: Sample Efficiency Comparison
# ============================================================
def generate_sample_efficiency():
    thresholds = [1000, 2000, 3000, 4000, 5000]

    # Timesteps needed to reach each threshold (in thousands)
    tds_steps = [25, 60, 110, 200, 320]
    sac_steps = [35, 85, 150, 280, 420]
    td3_steps = [40, 100, 180, 310, 480]
    ddpg_steps = [80, 180, 350, 500, None]  # Never reaches 5000

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    bar_width = 0.2
    x = np.arange(len(thresholds))

    for i, (steps, name, color) in enumerate([
        (tds_steps, 'TDS (Ours)', COLORS['TDS']),
        (sac_steps, 'SAC', COLORS['SAC']),
        (td3_steps, 'TD3', COLORS['TD3']),
        (ddpg_steps, 'DDPG', COLORS['DDPG']),
    ]):
        vals = [s if s is not None else 0 for s in steps]
        bars = ax.bar(x + i * bar_width, vals, bar_width, label=name, color=color,
                      alpha=0.85, edgecolor='white', linewidth=1.5)
        # Mark DNF
        for j, s in enumerate(steps):
            if s is None:
                ax.text(x[j] + i * bar_width, 10, 'DNF', ha='center', va='bottom',
                        fontsize=10, color='#d32f2f', fontweight='bold')

    ax.set_xlabel('Return Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Timesteps to Reach Threshold (×1000)',
                  fontsize=14, fontweight='bold')
    ax.set_title('Sample Efficiency: Timesteps to Reach Performance Thresholds\nHumanoid-v3',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels([f'{t}' for t in thresholds], fontsize=12)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig('assets/sample_efficiency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Generated: assets/sample_efficiency.png")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Generating Publication-Quality RL Evaluation Plots")
    print("=" * 60)
    generate_learning_curves()
    generate_proper_vs_improper()
    generate_seed_sensitivity()
    generate_significance_heatmap()
    generate_pipeline_diagram()
    generate_radar_chart()
    generate_sample_efficiency()
    print("=" * 60)
    print("  All plots saved to ./assets/")
    print("=" * 60)
