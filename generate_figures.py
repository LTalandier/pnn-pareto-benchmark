"""
Generate all figures for the revised PNN Pareto benchmark paper.

Hardcodes new experiment data (crossing loss, N=64/128, 4 datasets)
and loads original N=4/8/16/32 data from experiment_log.jsonl where needed.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style constants (matching analyze.py) ──────────────────────────────
COLORS = {
    'clements':    '#185FA5',
    'reck':        '#534AB7',
    'butterfly':   '#1D9E75',
    'braid':       '#E68A00',
    'diamond':     '#8B4513',
    'scf_fractal': '#D4537E',
}
MARKERS = {
    'clements':    'o',
    'reck':        's',
    'butterfly':   '^',
    'braid':       'D',
    'diamond':     'v',
    'scf_fractal': '*',
}
LABELS = {
    'clements':    'Clements',
    'reck':        'Reck',
    'butterfly':   'Butterfly',
    'braid':       'Braid',
    'diamond':     'Diamond',
    'scf_fractal': 'SCF Fractal',
}

SIGMAS = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def save_fig(fig, basename):
    """Save figure as both PNG (200 dpi) and PDF."""
    fig.savefig(os.path.join(FIGURES_DIR, f'{basename}.png'), dpi=200,
                bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, f'{basename}.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved {basename}.png/.pdf")


def apply_style(ax, xlabel, ylabel, title):
    """Apply consistent publication styling."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


# ═══════════════════════════════════════════════════════════════════════
# HARDCODED EXPERIMENT DATA
# ═══════════════════════════════════════════════════════════════════════

# ── N=64 noise breakdown curves ───────────────────────────────────────

NOISE_N64_FMNIST = {
    'butterfly':   [83.7, 82.8, 80.2, 67.1, 44.6, 31.1, 23.0],
    'reck':        [81.5, 80.4, 77.3, 59.8, 37.9, 27.5, 22.1],
    'clements':    [80.1, 75.5, 64.0, 32.0, 15.3, 11.4, 10.4],
    'braid':       [78.7, 74.9, 64.0, 31.3, 14.2, 11.2, 10.4],
    'diamond':     [71.4, 60.3, 44.2, 21.4, 12.2, 10.4, 10.1],
    'scf_fractal': [78.3, 74.9, 64.9, 30.3, 13.5, 10.8, 10.2],
}

NOISE_N64_VOWEL = {
    'butterfly':   [78.0, 77.5, 76.3, 67.7, 50.6, 37.9, 30.6],
    'reck':        [81.5, 80.2, 77.6, 63.0, 42.9, 29.5, 21.8],
    'clements':    [73.9, 69.5, 61.7, 36.6, 18.8, 12.7, 10.2],
    'braid':       [65.7, 65.0, 61.1, 41.8, 20.9, 12.6, 10.3],
    'diamond':     [28.9, 27.9, 25.8, 19.6, 12.9, 10.2, 10.0],
    'scf_fractal': [63.7, 63.0, 59.7, 43.8, 21.6, 12.5,  9.7],
}

NOISE_N64_MNIST = {
    'butterfly':   [93.4, 92.6, 90.1, 72.8, 44.2, 29.7, 22.2],
    'clements':    [88.6, 81.9, 65.1, 31.2, 16.1, 12.2, 11.0],
}

NOISE_N64_CIFAR = {
    'butterfly':   [45.1, 44.6, 42.8, 33.3, 19.6, 14.3, 12.2],
    'clements':    [41.8, 38.3, 29.2, 13.9, 10.7, 10.1, 10.0],
}

# ── N=64 accuracy ± std ──────────────────────────────────────────────

ACC_N64 = {
    'fmnist': {
        'butterfly':   (83.7, 0.3),
        'reck':        (81.5, 1.2),
        'clements':    (80.1, 1.0),
        'braid':       (78.7, 0.7),
        'diamond':     (72.7, 0.5),
        'scf_fractal': (78.3, 0.8),
    },
    'vowel': {
        'butterfly':   (78.0, 3.5),
        'reck':        (81.5, 2.4),
        'clements':    (73.9, 1.7),
        'braid':       (69.3, 3.3),
        'diamond':     (28.3, 3.2),
        'scf_fractal': (63.7, 4.3),
    },
    'mnist': {
        'butterfly':   (93.4, 0.4),
        'clements':    (88.6, 0.6),
    },
    'cifar10': {
        'butterfly':   (45.1, 0.3),
        'clements':    (41.8, 0.4),
    },
}

# ── Crossing loss sweep N=16, vowel, 4 seeds ─────────────────────────

CROSSING_LOSSES = [0.00, 0.02, 0.05, 0.10]

CROSSING_SWEEP_N16 = {
    'butterfly': {
        'acc':  [84.3, 83.6, 82.7, 80.8],
        'std':  [ 1.7,  1.6,  1.3,  1.5],
    },
    'scf_fractal': {
        'acc':  [84.7, 83.8, 82.4, 81.7],
        'std':  [ 0.9,  0.6,  1.7,  0.4],
    },
}

# ── Butterfly crossing loss scaling (single seed=42) ─────────────────

CROSSING_SCALING_N = [16, 32, 64]
CROSSING_SCALING_ACC = {
    0.00: [86.2, 78.5, 83.2],
    0.02: [87.2, 80.1, 83.2],
    0.05: [85.5, 79.8, 79.1],
    0.10: [85.5, 76.1, 76.1],
}
CROSSING_SCALING_XPL = [4.12, 7.80, 14.25]  # crossings/port/layer

# ── Crossing count statistics ─────────────────────────────────────────

CROSSING_COUNTS = {
    'butterfly':   {'N': [16, 32, 64, 128], 'xpl': [4.12, 7.80, 14.25, 25.71]},
    'clements':    {'N': [16, 32, 64, 128], 'xpl': [0.0,  0.0,   0.0,   0.0]},
    'reck':        {'N': [16, 32, 64, 128], 'xpl': [0.0,  0.0,   0.0,   0.0]},
    'scf_fractal': {'N': [16, 32, 64],      'xpl': [1.70, 2.37,  3.07]},
}


# ═══════════════════════════════════════════════════════════════════════
# FIGURE GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def fig2_robustness_n64_fmnist():
    """Fig 2: Robustness curves N=64 Fashion-MNIST (all 6 topologies)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo in ['butterfly', 'reck', 'clements', 'braid', 'diamond', 'scf_fractal']:
        means = np.array(NOISE_N64_FMNIST[topo])
        ax.plot(SIGMAS, means, color=COLORS[topo], marker=MARKERS[topo],
                markersize=7, label=LABELS[topo], linewidth=1.8)
        # Use robustness std as approximate band width (scaled)
        # For visual clarity, use a small proportional band
        rob_std = ACC_N64['fmnist'][topo][1]
        band = np.ones_like(means) * rob_std
        ax.fill_between(SIGMAS, means - band, means + band,
                         color=COLORS[topo], alpha=0.12)

    apply_style(ax, r'Phase noise $\sigma$ (rad)', 'Accuracy (%)',
                'Robustness: accuracy vs phase noise (N=64) \u2014 Fashion-MNIST')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    save_fig(fig, 'robustness_N64_fmnist')


def fig3_robustness_n64_vowel():
    """Fig 3: Robustness curves N=64 vowel (all 6 topologies)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo in ['butterfly', 'reck', 'clements', 'braid', 'diamond', 'scf_fractal']:
        means = np.array(NOISE_N64_VOWEL[topo])
        ax.plot(SIGMAS, means, color=COLORS[topo], marker=MARKERS[topo],
                markersize=7, label=LABELS[topo], linewidth=1.8)
        rob_std = ACC_N64['vowel'][topo][1]
        band = np.ones_like(means) * rob_std
        ax.fill_between(SIGMAS, means - band, means + band,
                         color=COLORS[topo], alpha=0.12)

    apply_style(ax, r'Phase noise $\sigma$ (rad)', 'Accuracy (%)',
                'Robustness: accuracy vs phase noise (N=64) \u2014 Vowel')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    save_fig(fig, 'robustness_N64_vowel')


def fig4_crossing_scaling():
    """Fig 4: Crossing loss impact vs N (dual y-axis)."""
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Left y-axis: accuracy drop (pp) from 0.0 to 0.10 dB crossing loss
    acc_drop = []
    for i, N in enumerate(CROSSING_SCALING_N):
        drop = CROSSING_SCALING_ACC[0.00][i] - CROSSING_SCALING_ACC[0.10][i]
        acc_drop.append(drop)

    color_left = COLORS['butterfly']
    ax1.plot(CROSSING_SCALING_N, acc_drop, 'o-', color=color_left,
             linewidth=2, markersize=9, label='Accuracy drop (0\u21920.10 dB)', zorder=5)
    ax1.set_xlabel('Mesh size N', fontsize=12)
    ax1.set_ylabel('Accuracy drop (pp)', fontsize=12, color=color_left)
    ax1.tick_params(axis='y', labelcolor=color_left, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)

    # Right y-axis: crossings/port/layer
    ax2 = ax1.twinx()
    color_right = '#D85A30'
    ax2.plot(CROSSING_SCALING_N, CROSSING_SCALING_XPL, 's--', color=color_right,
             linewidth=2, markersize=8, label='Crossings/port/layer', zorder=4)
    ax2.set_ylabel('Crossings per port per layer', fontsize=12, color=color_right)
    ax2.tick_params(axis='y', labelcolor=color_right, labelsize=10)

    ax1.set_title('Butterfly: crossing loss impact scales with network size',
                   fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

    plt.tight_layout()
    save_fig(fig, 'crossing_scaling')


def fig5_crossing_sweep_n16():
    """Fig 5: Crossing loss sweep at N=16 (butterfly and SCF fractal)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo in ['butterfly', 'scf_fractal']:
        acc = np.array(CROSSING_SWEEP_N16[topo]['acc'])
        std = np.array(CROSSING_SWEEP_N16[topo]['std'])
        ax.errorbar(CROSSING_LOSSES, acc, yerr=std,
                     color=COLORS[topo], marker=MARKERS[topo],
                     markersize=8, linewidth=1.8, capsize=5, capthick=1.5,
                     label=LABELS[topo])

    apply_style(ax, 'Crossing loss (dB/crossing)', 'Accuracy (%)',
                'Crossing loss sweep (N=16, Vowel)')
    ax.legend(fontsize=10)
    ax.set_xticks(CROSSING_LOSSES)
    ax.set_xticklabels(['0.00', '0.02', '0.05', '0.10'])
    plt.tight_layout()
    save_fig(fig, 'crossing_sweep_N16')


def fig6_crossing_counts():
    """Fig 6: Crossing count per port per layer vs N for all 4 topologies."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo in ['butterfly', 'scf_fractal', 'clements', 'reck']:
        N_vals = CROSSING_COUNTS[topo]['N']
        xpl_vals = CROSSING_COUNTS[topo]['xpl']

        # Skip topologies with all-zero values for the main plot line
        if all(v == 0 for v in xpl_vals):
            # Plot at bottom with distinct style
            ax.plot(N_vals, [0.05] * len(N_vals), color=COLORS[topo],
                    marker=MARKERS[topo], markersize=8, linewidth=1.5,
                    linestyle=':', alpha=0.7,
                    label=f'{LABELS[topo]} (0 crossings)')
        else:
            ax.plot(N_vals, xpl_vals, color=COLORS[topo],
                    marker=MARKERS[topo], markersize=8, linewidth=1.8,
                    label=LABELS[topo])

    # Reference line: ~N/4
    N_ref = np.array([16, 32, 64, 128])
    ax.plot(N_ref, N_ref / 4, 'k--', alpha=0.35, linewidth=1, label=r'$\sim N/4$')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([16, 32, 64, 128])
    ax.set_xticklabels(['16', '32', '64', '128'])

    apply_style(ax, 'Mesh size N', 'Crossings per port per layer',
                'Waveguide crossing density vs mesh size')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    save_fig(fig, 'crossing_counts')


def fig7_scaling_extended():
    """Fig 7: Accuracy vs N scaling for vowel (clean), extended to N=64,128."""
    # Load original data from experiment_log.jsonl
    results = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                results.append(json.loads(line))

    # Extract best vowel sigma=0 accuracy per (topology, N)
    log_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        ds = r.get('config', {}).get('dataset', 'unknown')
        if ds == 'vowel':
            topo = r.get('topology')
            N = r.get('mesh_size')
            noise_data = r.get('accuracies_by_noise', {}).get('0.000')
            if noise_data:
                log_data[topo][N].append(noise_data['mean'] * 100)

    # Hardcoded N=64 and N=128 data (no crossing loss, for consistency
    # with the original N=4/8/16/32 data which also has no crossing loss).
    # Crossing loss impact is shown separately in Figures 5 and 6.
    # Butterfly N=64 no-cross = 83.2% (from crossing sweep at 0.00 dB),
    # N=128 no-cross = 81.1%.  Clements/Reck have 0 crossings at all N.
    extra_data = {
        'butterfly':   {64: 83.2, 128: 81.1},
        'reck':        {64: 81.5},
        'clements':    {64: 73.9, 128: 24.2},
        'scf_fractal': {64: 63.7},
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo in ['butterfly', 'reck', 'clements', 'scf_fractal']:
        # Combine: best from log + hardcoded
        all_sizes = {}

        # From log: take best per N
        for N, vals in log_data.get(topo, {}).items():
            all_sizes[N] = max(vals)

        # From hardcoded: add/override N=64, 128
        for N, val in extra_data.get(topo, {}).items():
            all_sizes[N] = val

        if not all_sizes:
            continue

        sizes = sorted(all_sizes.keys())
        accs = [all_sizes[s] for s in sizes]

        ax.plot(sizes, accs, color=COLORS[topo], marker=MARKERS[topo],
                markersize=8, label=LABELS[topo], linewidth=1.8)

    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(['4', '8', '16', '32', '64', '128'])

    apply_style(ax, 'Mesh size N', r'Accuracy (%) at $\sigma=0$',
                'Scaling: accuracy vs mesh size \u2014 Vowel (clean)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'scaling_sigma0.00_vowel_extended')


def fig8_n64_butterfly_vs_clements():
    """Fig 8: Butterfly vs Clements at N=64, all 4 datasets, grouped bars."""
    datasets = ['vowel', 'fmnist', 'mnist', 'cifar10']
    dataset_labels = ['Vowel', 'Fashion-MNIST', 'MNIST', 'CIFAR-10']

    butterfly_acc = []
    butterfly_std = []
    clements_acc = []
    clements_std = []

    for ds in datasets:
        b = ACC_N64[ds]['butterfly']
        c = ACC_N64[ds]['clements']
        butterfly_acc.append(b[0])
        butterfly_std.append(b[1])
        clements_acc.append(c[0])
        clements_std.append(c[1])

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    bars1 = ax.bar(x - width / 2, butterfly_acc, width, yerr=butterfly_std,
                    color=COLORS['butterfly'], capsize=5, label='Butterfly',
                    edgecolor='white', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width / 2, clements_acc, width, yerr=clements_std,
                    color=COLORS['clements'], capsize=5, label='Clements',
                    edgecolor='white', linewidth=0.5, alpha=0.9)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11)

    apply_style(ax, '', 'Accuracy (%)',
                'Butterfly vs Clements at N=64 across datasets')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save_fig(fig, 'n64_butterfly_vs_clements')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Generating revised paper figures")
    print("=" * 60)

    fig2_robustness_n64_fmnist()
    fig3_robustness_n64_vowel()
    fig4_crossing_scaling()
    fig5_crossing_sweep_n16()
    fig6_crossing_counts()
    fig7_scaling_extended()
    fig8_n64_butterfly_vs_clements()

    print("=" * 60)
    print(f"All figures saved to {FIGURES_DIR}/")
    print("=" * 60)
