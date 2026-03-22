"""
Generate fmnist Pareto plots with error bars where multi-seed data available (N=4,8).
N=16 remains single-run.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

os.chdir("/root/pnn-pareto")

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TOPOLOGY_COLORS = {
    'clements': '#185FA5', 'reck': '#534AB7',
    'butterfly': '#1D9E75', 'scf_fractal': '#D4537E',
}
TOPOLOGY_MARKERS = {
    'clements': 'o', 'reck': 's', 'butterfly': '^', 'scf_fractal': '*',
}
TOPOLOGY_LABELS = {
    'clements': 'Clements', 'reck': 'Reck',
    'butterfly': 'Butterfly', 'scf_fractal': 'SCF Fractal',
}

results = []
with open(LOG_FILE) as f:
    for line in f:
        results.append(json.loads(line))

def get_fmnist_grouped():
    grouped = defaultdict(list)
    for r in results:
        c = r.get("config", {})
        if c.get("dataset") != "fmnist": continue
        if abs(c.get("loss_per_mzi_dB", 0.2) - 0.2) > 0.01: continue
        if c.get("noise_aware_training", False): continue
        if c.get("n_photonic_layers", 1) != 1: continue
        grouped[(c["topology"], c["mesh_size"])].append(r)
    return grouped

def plot_pareto_fmnist(noise_sigma=0.0):
    sigma_key = f'{noise_sigma:.3f}'
    grouped = get_fmnist_grouped()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    all_costs, all_accs = [], []

    for topo in TOPOLOGY_COLORS:
        costs, accs, stds, sizes = [], [], [], []
        for N in sorted(set(k[1] for k in grouped if k[0] == topo)):
            runs = grouped.get((topo, N), [])
            if not runs: continue
            hw_cost = runs[0]["hardware_cost"]
            run_accs = [r["accuracies_by_noise"].get(sigma_key, {}).get("mean") for r in runs]
            run_accs = [a for a in run_accs if a is not None]
            if not run_accs: continue
            costs.append(hw_cost)
            accs.append(np.mean(run_accs))
            stds.append(np.std(run_accs) if len(run_accs) > 1 else 0)
            sizes.append(N)

        if not costs: continue
        ax.errorbar(costs, accs, yerr=stds,
            color=TOPOLOGY_COLORS[topo], marker=TOPOLOGY_MARKERS[topo],
            markersize=8, capsize=4, capthick=1.5, linewidth=0, elinewidth=1.5,
            label=TOPOLOGY_LABELS[topo], alpha=0.9, markeredgecolor='white', markeredgewidth=0.5)
        for c, a, s in zip(costs, accs, sizes):
            ax.annotate(f'N={s}', (c, a), fontsize=11, fontweight='bold',
                textcoords='offset points', xytext=(7, 7),
                color=TOPOLOGY_COLORS[topo], alpha=0.85)
        all_costs.extend(costs)
        all_accs.extend(accs)

    if all_costs:
        ca, aa = np.array(all_costs), np.array(all_accs)
        n = len(ca)
        opt = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and ca[j] <= ca[i] and aa[j] >= aa[i] and (ca[j] < ca[i] or aa[j] > aa[i]):
                    opt[i] = False; break
        si = np.argsort(ca[opt])
        ax.plot(ca[opt][si], aa[opt][si], 'k--', alpha=0.4, linewidth=1, label='Pareto front')

    sigma_label = f'σ={noise_sigma} rad' if noise_sigma > 0 else 'clean'
    ax.set_xlabel('Hardware cost (MZIs × depth)', fontsize=12)
    ax.set_ylabel(f'Accuracy ({sigma_label})', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('PNN topology Pareto front — fmnist', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'pareto_sigma{noise_sigma:.2f}_fmnist'
    fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.png'), dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f'{fname}.pdf'))
    plt.close()
    print(f"[PLOT] Saved {fname}")

for sigma in [0.0, 0.05, 0.1, 0.2]:
    plot_pareto_fmnist(sigma)
print("Done.")
