"""
Generate vowel Pareto and scaling plots with mean +/- std error bars.
Reads experiment_log.jsonl, groups by (topology, mesh_size, loss_dB),
computes mean/std over seeds.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

os.chdir("/root/pnn-pareto")

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TOPOLOGY_COLORS = {
    'clements': '#185FA5',
    'reck': '#534AB7',
    'butterfly': '#1D9E75',
    'scf_fractal': '#D4537E',
}
TOPOLOGY_MARKERS = {
    'clements': 'o',
    'reck': 's',
    'butterfly': '^',
    'scf_fractal': '*',
}
TOPOLOGY_LABELS = {
    'clements': 'Clements',
    'reck': 'Reck',
    'butterfly': 'Butterfly',
    'scf_fractal': 'SCF Fractal',
}

def load_results():
    results = []
    with open(LOG_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_vowel_grouped(results, loss_dB=0.2):
    """Group vowel results by (topology, mesh_size), return lists of runs."""
    grouped = defaultdict(list)
    for r in results:
        c = r.get("config", {})
        if c.get("dataset") != "vowel":
            continue
        if abs(c.get("loss_per_mzi_dB", 0.2) - loss_dB) > 0.01:
            continue
        if c.get("noise_aware_training", False):
            continue
        key = (c["topology"], c["mesh_size"])
        grouped[key].append(r)
    return grouped


def plot_pareto_errorbars(results, noise_sigma=0.0, loss_dB=0.2, save=True):
    """Pareto front with error bars from multi-seed runs."""
    sigma_key = f'{noise_sigma:.3f}'
    grouped = get_vowel_grouped(results, loss_dB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    all_costs, all_accs = [], []

    for topo_name in TOPOLOGY_COLORS:
        costs_mean, accs_mean, accs_std, sizes = [], [], [], []

        for N in sorted(set(k[1] for k in grouped if k[0] == topo_name)):
            runs = grouped.get((topo_name, N), [])
            if not runs:
                continue

            # All runs have same hardware cost
            hw_cost = runs[0]["hardware_cost"]

            run_accs = []
            for r in runs:
                noise_data = r.get("accuracies_by_noise", {}).get(sigma_key)
                if noise_data:
                    run_accs.append(noise_data["mean"])

            if not run_accs:
                continue

            mean_acc = np.mean(run_accs)
            std_acc = np.std(run_accs) if len(run_accs) > 1 else 0

            costs_mean.append(hw_cost)
            accs_mean.append(mean_acc)
            accs_std.append(std_acc)
            sizes.append(N)

        if not costs_mean:
            continue

        costs_mean = np.array(costs_mean)
        accs_mean = np.array(accs_mean)
        accs_std = np.array(accs_std)

        ax.errorbar(costs_mean, accs_mean, yerr=accs_std,
                     color=TOPOLOGY_COLORS[topo_name],
                     marker=TOPOLOGY_MARKERS[topo_name],
                     markersize=8, capsize=4, capthick=1.5,
                     linewidth=0, elinewidth=1.5,
                     label=TOPOLOGY_LABELS[topo_name],
                     alpha=0.9, markeredgecolor='white', markeredgewidth=0.5)

        for c, a, s in zip(costs_mean, accs_mean, sizes):
            ax.annotate(f'N={s}', (c, a), fontsize=11, fontweight='bold',
                        textcoords='offset points', xytext=(7, 7),
                        color=TOPOLOGY_COLORS[topo_name], alpha=0.85)

        all_costs.extend(costs_mean)
        all_accs.extend(accs_mean)

    # Pareto front from means
    if all_costs:
        costs_arr = np.array(all_costs)
        accs_arr = np.array(all_accs)
        n = len(costs_arr)
        is_optimal = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if costs_arr[j] <= costs_arr[i] and accs_arr[j] >= accs_arr[i]:
                    if costs_arr[j] < costs_arr[i] or accs_arr[j] > accs_arr[i]:
                        is_optimal[i] = False
                        break
        pareto_costs = costs_arr[is_optimal]
        pareto_accs = accs_arr[is_optimal]
        sort_idx = np.argsort(pareto_costs)
        ax.plot(pareto_costs[sort_idx], pareto_accs[sort_idx],
                'k--', alpha=0.4, linewidth=1, label='Pareto front')

    ax.set_xlabel('Hardware cost (MZIs × depth)', fontsize=12)
    sigma_label = f'σ={noise_sigma} rad' if noise_sigma > 0 else 'clean'
    ax.set_ylabel(f'Accuracy ({sigma_label})', fontsize=12)
    ax.set_xscale('log')
    ax.set_title(f'PNN topology Pareto front — vowel', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'pareto_sigma{noise_sigma:.2f}_vowel'
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
        plt.close()
        print(f"[PLOT] Saved {fname}.png/.pdf")
    return fig


def plot_scaling_errorbars(results, noise_sigma=0.0, save=True):
    """Scaling plot with error bars."""
    sigma_key = f'{noise_sigma:.3f}'
    grouped = get_vowel_grouped(results, loss_dB=0.2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for topo_name in TOPOLOGY_COLORS:
        sizes_list, means_list, stds_list = [], [], []

        for N in sorted(set(k[1] for k in grouped if k[0] == topo_name)):
            runs = grouped.get((topo_name, N), [])
            if not runs:
                continue

            run_accs = []
            for r in runs:
                noise_data = r.get("accuracies_by_noise", {}).get(sigma_key)
                if noise_data:
                    run_accs.append(noise_data["mean"])

            if not run_accs:
                continue

            sizes_list.append(N)
            means_list.append(np.mean(run_accs))
            stds_list.append(np.std(run_accs) if len(run_accs) > 1 else 0)

        if not sizes_list:
            continue

        sizes_arr = np.array(sizes_list)
        means_arr = np.array(means_list)
        stds_arr = np.array(stds_list)

        ax.errorbar(sizes_arr, means_arr, yerr=stds_arr,
                     color=TOPOLOGY_COLORS[topo_name],
                     marker=TOPOLOGY_MARKERS[topo_name],
                     markersize=8, capsize=4, capthick=1.5,
                     linewidth=1.5, elinewidth=1.5,
                     label=TOPOLOGY_LABELS[topo_name])

    ax.set_xlabel('Mesh size N', fontsize=12)
    sigma_label = f'σ={noise_sigma} rad' if noise_sigma > 0 else 'clean'
    ax.set_ylabel(f'Accuracy ({sigma_label})', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_title(f'Scaling: accuracy vs mesh size — vowel', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'scaling_sigma{noise_sigma:.2f}_vowel'
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
        plt.close()
        print(f"[PLOT] Saved {fname}.png/.pdf")
    return fig


if __name__ == '__main__':
    results = load_results()
    print(f"Loaded {len(results)} experiments")

    # Pareto plots for vowel
    for sigma in [0.0, 0.05, 0.1, 0.2]:
        plot_pareto_errorbars(results, noise_sigma=sigma)

    # Scaling plots for vowel
    for sigma in [0.0, 0.1]:
        plot_scaling_errorbars(results, noise_sigma=sigma)

    print("Done.")
