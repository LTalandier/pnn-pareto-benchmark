"""
Crossing loss sweep: butterfly + scf_fractal at N=16 vowel, Lx=0.00/0.02/0.05/0.10.
Uses the SAME seeds as Tables 3/9: {no-seed, 1, 2, 3}.

For Lx=0.00, we reuse the existing experiment log data (ids 10,42,43,44 for butterfly;
11,45,46,47 for scf_fractal) since those were run with crossing_loss_dB=0.0 by default.

For Lx=0.02/0.05/0.10, we rerun with crossing_loss_dB set explicitly.
"""
import torch
import numpy as np
import json
import os

from physics import PhotonicNeuralNetwork
from prepare import prepare_data
from evaluate import evaluate_model, train_model

TOPOLOGIES = ["butterfly", "scf_fractal"]
CROSSING_LOSSES = [0.02, 0.05, 0.10]  # skip 0.00 — use existing data
# Seeds matching Tables 3/9: None (no seed), 1, 2, 3
SEEDS = [None, 1, 2, 3]

MESH_SIZE = 16
DATASET = "vowel"
LOSS_PER_MZI_DB = 0.2
N_EPOCHS = 100
LR = 0.005

results = {}

for topo in TOPOLOGIES:
    results[topo] = {}
    for lx in CROSSING_LOSSES:
        accs = []
        robs = []
        for seed in SEEDS:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            else:
                # Reset to non-deterministic (matching the original default run)
                torch.seed()  # random seed
                np.random.seed()

            device = torch.device("cpu")
            train_loader, test_loader, data_info = prepare_data(DATASET, pca_dim=MESH_SIZE)
            n_classes = data_info["n_classes"]

            model = PhotonicNeuralNetwork(
                N=MESH_SIZE, topology=topo, n_classes=n_classes,
                n_photonic_layers=1, loss_per_mzi_dB=LOSS_PER_MZI_DB,
                classifier_hidden=(64,), nonlinearity="photodetect",
                crossing_loss_dB=lx,
            ).to(device)
            model.N = MESH_SIZE
            model.topology_name = topo

            train_time = train_model(model, train_loader, device,
                                     n_epochs=N_EPOCHS, lr=LR)
            res = evaluate_model(model, test_loader, device)

            acc = res["clean_accuracy"] * 100
            rob = res["robustness"]
            accs.append(acc)
            robs.append(rob)
            tag = f"seed={seed}" if seed is not None else "seed=default"
            print(f"  {topo} Lx={lx:.2f} {tag}: acc={acc:.2f}% rob={rob:.4f}")

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=0)  # population std to match paper convention
        mean_rob = np.mean(robs)
        std_rob = np.std(robs, ddof=0)
        results[topo][f"{lx:.2f}"] = {
            "accs": accs,
            "robs": robs,
            "mean_acc": round(mean_acc, 1),
            "std_acc": round(std_acc, 1),
            "mean_rob": round(mean_rob, 3),
            "std_rob": round(std_rob, 3),
        }
        print(f"  >> {topo} Lx={lx:.2f}: {mean_acc:.1f}±{std_acc:.1f}%, rob={mean_rob:.3f}±{std_rob:.3f}")

# Also compute Lx=0.00 from existing experiment log data
print("\n--- Lx=0.00 from existing experiment log ---")
with open("results/experiment_log.jsonl") as f:
    log = [json.loads(line) for line in f]

for topo, ids in [("butterfly", [10, 42, 43, 44]), ("scf_fractal", [11, 45, 46, 47])]:
    accs = []
    robs = []
    for eid in ids:
        e = log[eid]
        assert e["id"] == eid
        accs.append(e["clean_accuracy"] * 100)
        robs.append(e["robustness"])
    mean_acc = np.mean(accs)
    std_acc = np.std(accs, ddof=0)
    mean_rob = np.mean(robs)
    std_rob = np.std(robs, ddof=0)
    results[topo]["0.00"] = {
        "accs": accs,
        "robs": robs,
        "mean_acc": round(mean_acc, 1),
        "std_acc": round(std_acc, 1),
        "mean_rob": round(mean_rob, 3),
        "std_rob": round(std_rob, 3),
    }
    print(f"  {topo} Lx=0.00: {mean_acc:.1f}±{std_acc:.1f}%, rob={mean_rob:.3f}±{std_rob:.3f}")

# Summary table
print("\n" + "="*70)
print("SUMMARY — Table 7 replacement values")
print("="*70)
for topo in TOPOLOGIES:
    print(f"\n{topo}:")
    for lx in ["0.00", "0.02", "0.05", "0.10"]:
        d = results[topo][lx]
        print(f"  Lx={lx}: {d['mean_acc']}±{d['std_acc']}%, rob={d['mean_rob']}±{d['std_rob']}")

# Save raw results
with open("results/crossing_sweep_canonical.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/crossing_sweep_canonical.json")
