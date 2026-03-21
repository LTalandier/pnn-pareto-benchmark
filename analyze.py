"""
Pareto front analysis and plot generation.
FIXED — the agent must never modify this file.

Generates publication-quality figures for the benchmark paper.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TOPOLOGY_COLORS = {
    'clements': '#185FA5',
    'reck': '#534AB7',
    'butterfly': '#1D9E75',
    'diamond': '#D85A30',
    'braid': '#BA7517',
    'scf_fractal': '#D4537E',
}
TOPOLOGY_MARKERS = {
    'clements': 'o',
    'reck': 's',
    'butterfly': '^',
    'diamond': 'D',
    'braid': 'P',
    'scf_fractal': '*',
}


def load_results():
    results = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                results.append(json.loads(line))
    return results


def is_pareto_optimal(costs, accuracies):
    """
    Find Pareto-optimal points (minimize cost, maximize accuracy).
    Returns boolean mask.
    """
    n = len(costs)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower cost AND higher accuracy
            if costs[j] <= costs[i] and accuracies[j] >= accuracies[i]:
                if costs[j] < costs[i] or accuracies[j] > accuracies[i]:
                    is_optimal[i] = False
                    break
    return is_optimal


def plot_pareto_front(results=None, noise_sigma=0.0, dataset=None,
                      mesh_sizes=None, save=True):
    """
    Plot accuracy vs hardware cost, colored by topology,
    with Pareto front highlighted.
    """
    if results is None:
        results = load_results()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    sigma_key = f'{noise_sigma:.3f}'
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    all_costs, all_accs, all_labels = [], [], []

    for topo_name in TOPOLOGY_COLORS:
        topo_results = [r for r in results
                        if r.get('topology') == topo_name
                        and (dataset is None or r.get('config', {}).get('dataset') == dataset)
                        and (mesh_sizes is None or r.get('mesh_size') in mesh_sizes)]

        costs, accs, sizes = [], [], []
        for r in topo_results:
            noise_data = r.get('accuracies_by_noise', {}).get(sigma_key)
            if noise_data is None:
                continue
            costs.append(r['hardware_cost'])
            accs.append(noise_data['mean'])
            sizes.append(r.get('mesh_size', 0))

        if not costs:
            continue

        ax.scatter(costs, accs,
                   c=TOPOLOGY_COLORS[topo_name],
                   marker=TOPOLOGY_MARKERS[topo_name],
                   s=80, alpha=0.8, label=topo_name,
                   edgecolors='white', linewidths=0.5)

        # Annotate with mesh size
        for c, a, s in zip(costs, accs, sizes):
            ax.annotate(f'N={s}', (c, a), fontsize=7,
                        textcoords='offset points', xytext=(5, 5),
                        color=TOPOLOGY_COLORS[topo_name], alpha=0.7)

        all_costs.extend(costs)
        all_accs.extend(accs)

    # Draw Pareto front
    if all_costs:
        costs_arr = np.array(all_costs)
        accs_arr = np.array(all_accs)
        pareto_mask = is_pareto_optimal(costs_arr, accs_arr)
        pareto_costs = costs_arr[pareto_mask]
        pareto_accs = accs_arr[pareto_mask]
        sort_idx = np.argsort(pareto_costs)
        ax.plot(pareto_costs[sort_idx], pareto_accs[sort_idx],
                'k--', alpha=0.4, linewidth=1, label='Pareto front')

    ax.set_xlabel('Hardware cost (MZIs × depth)', fontsize=12)
    ax.set_ylabel(f'Accuracy (σ={noise_sigma} rad)', fontsize=12)
    ax.set_xscale('log')
    title = 'PNN topology Pareto front'
    if dataset:
        title += f' — {dataset}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'pareto_sigma{noise_sigma:.2f}'
        if dataset:
            fname += f'_{dataset}'
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
        plt.close()
        print(f"[PLOT] Saved {fname}.png/.pdf")
    return fig


def plot_robustness_curves(results=None, dataset=None, mesh_size=None, save=True):
    """
    Plot accuracy vs noise sigma for each topology at a fixed mesh size.
    """
    if results is None:
        results = load_results()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo_name in TOPOLOGY_COLORS:
        topo_results = [r for r in results
                        if r.get('topology') == topo_name
                        and (dataset is None or r.get('config', {}).get('dataset') == dataset)
                        and (mesh_size is None or r.get('mesh_size') == mesh_size)]

        if not topo_results:
            continue

        # Take the best run (highest clean accuracy) for this topology
        best = max(topo_results, key=lambda r: r.get('clean_accuracy', 0))
        noise_data = best.get('accuracies_by_noise', {})

        sigmas, means, stds = [], [], []
        for sigma_key in sorted(noise_data.keys(), key=float):
            sigmas.append(float(sigma_key))
            means.append(noise_data[sigma_key]['mean'])
            stds.append(noise_data[sigma_key].get('std', 0))

        if sigmas:
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(sigmas, means, color=TOPOLOGY_COLORS[topo_name],
                    marker=TOPOLOGY_MARKERS[topo_name], markersize=6,
                    label=topo_name, linewidth=1.5)
            ax.fill_between(sigmas, means - stds, means + stds,
                            color=TOPOLOGY_COLORS[topo_name], alpha=0.1)

    ax.set_xlabel('Phase noise σ (rad)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    title = 'Robustness: accuracy vs phase noise'
    if mesh_size:
        title += f' (N={mesh_size})'
    if dataset:
        title += f' — {dataset}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'robustness_N{mesh_size or "all"}'
        if dataset:
            fname += f'_{dataset}'
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
        plt.close()
        print(f"[PLOT] Saved {fname}.png/.pdf")
    return fig


def plot_scaling(results=None, dataset=None, noise_sigma=0.0, save=True):
    """
    Plot accuracy vs mesh size N for each topology.
    """
    if results is None:
        results = load_results()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    sigma_key = f'{noise_sigma:.3f}'
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo_name in TOPOLOGY_COLORS:
        topo_results = [r for r in results
                        if r.get('topology') == topo_name
                        and (dataset is None or r.get('config', {}).get('dataset') == dataset)]

        size_acc = defaultdict(list)
        for r in topo_results:
            noise_data = r.get('accuracies_by_noise', {}).get(sigma_key)
            if noise_data:
                size_acc[r['mesh_size']].append(noise_data['mean'])

        if not size_acc:
            continue

        sizes = sorted(size_acc.keys())
        means = [max(size_acc[s]) for s in sizes]  # best run per size

        ax.plot(sizes, means, color=TOPOLOGY_COLORS[topo_name],
                marker=TOPOLOGY_MARKERS[topo_name], markersize=8,
                label=topo_name, linewidth=1.5)

    ax.set_xlabel('Mesh size N', fontsize=12)
    ax.set_ylabel(f'Accuracy (σ={noise_sigma} rad)', fontsize=12)
    ax.set_xscale('log', base=2)
    title = 'Scaling: accuracy vs mesh size'
    if dataset:
        title += f' — {dataset}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'scaling_sigma{noise_sigma:.2f}'
        if dataset:
            fname += f'_{dataset}'
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
        plt.close()
        print(f"[PLOT] Saved {fname}.png/.pdf")
    return fig


def generate_all_plots(datasets=None, mesh_sizes=None):
    """Generate all standard plots from the experiment log."""
    results = load_results()
    if not results:
        print("[PLOT] No results to plot.")
        return

    datasets = datasets or list(set(
        r.get('config', {}).get('dataset', 'unknown') for r in results))

    for ds in datasets:
        for sigma in [0.0, 0.05, 0.1, 0.2]:
            plot_pareto_front(results, noise_sigma=sigma, dataset=ds)

        sizes = sorted(set(r['mesh_size'] for r in results
                          if r.get('config', {}).get('dataset') == ds))
        for s in sizes:
            plot_robustness_curves(results, dataset=ds, mesh_size=s)

        plot_scaling(results, dataset=ds, noise_sigma=0.0)
        plot_scaling(results, dataset=ds, noise_sigma=0.1)

    print(f"[PLOT] Generated all plots in {FIGURES_DIR}/")


if __name__ == '__main__':
    generate_all_plots()
