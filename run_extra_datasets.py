"""
Additional dataset runs for Braid and Diamond at N=16 and N=64.
Datasets: Fashion-MNIST, MNIST, CIFAR-10 (Vowel already done in overnight suite).
"""

import json
import os
import time
import torch
import numpy as np

from physics import PhotonicNeuralNetwork
from prepare import prepare_data
from evaluate import train_model, evaluate_model

RESULTS_FILE = "results_overnight.json"
SEEDS = [None, 1, 2, 3]


def save_result(result):
    """Append one result to the JSON results file."""
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results.append(result)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_one(topology, N, dataset, seed, loss_dB=0.2, crossing_loss_dB=0.02):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.seed()
        np.random.seed()

    device = torch.device("cpu")
    bs = 1024 if dataset in ("fmnist", "mnist", "cifar10") else 128
    train_loader, test_loader, data_info = prepare_data(dataset, pca_dim=N, batch_size=bs)

    model = PhotonicNeuralNetwork(
        N=N, topology=topology, n_classes=data_info["n_classes"],
        n_photonic_layers=1, loss_per_mzi_dB=loss_dB,
        classifier_hidden=(64,), nonlinearity="photodetect",
        crossing_loss_dB=crossing_loss_dB,
    ).to(device)

    train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
    results = evaluate_model(model, test_loader, device)
    return results["clean_accuracy"] * 100, results["robustness"], train_time


if __name__ == "__main__":
    t_start = time.time()

    configs = []
    for topo in ["braid", "diamond"]:
        for N in [16, 64]:
            for ds in ["fmnist", "mnist", "cifar10"]:
                configs.append((topo, N, ds))

    total_runs = len(configs) * len(SEEDS)
    run_count = 0

    for topo, N, ds in configs:
        accs, robs = [], []
        for seed in SEEDS:
            t0 = time.time()
            cx_dB = 0.02 if topo == "braid" else 0.0
            acc, rob, _ = run_one(topo, N, ds, seed, crossing_loss_dB=cx_dB)
            elapsed = time.time() - t0
            accs.append(acc)
            robs.append(rob)
            run_count += 1
            stag = f"seed={seed}" if seed is not None else "seed=default"
            print(f"  [{run_count}/{total_runs}] {topo} N={N} {ds} {stag}: "
                  f"acc={acc:.1f}%, rob={rob:.3f} ({elapsed:.1f}s)")

        result = {
            "track": "extra-datasets",
            "topology": topo,
            "N": N,
            "dataset": ds,
            "epsilon": None,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=0)),
            "rob_mean": float(np.mean(robs)),
            "rob_std": float(np.std(robs, ddof=0)),
            "accs": accs,
            "robs": robs,
        }
        save_result(result)
        print(f"  >> {topo} N={N} {ds}: "
              f"{result['acc_mean']:.1f}+/-{result['acc_std']:.1f}%, "
              f"rob={result['rob_mean']:.3f}+/-{result['rob_std']:.3f}")
        print()

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")

    # Print summary table
    print("\n" + "=" * 70)
    print("EXTRA DATASET SUMMARY")
    print("=" * 70)

    # Load all results
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    extra = [r for r in all_results if r.get("track") == "extra-datasets"]
    for topo in ["braid", "diamond"]:
        print(f"\n{topo.upper()}:")
        print(f"{'N':>5} {'Dataset':<10} {'Acc (%)':>15} {'Rob':>15}")
        print("-" * 50)
        for r in sorted(
            [x for x in extra if x["topology"] == topo],
            key=lambda x: (x["N"], x["dataset"])
        ):
            print(f"{r['N']:>5} {r['dataset']:<10} "
                  f"{r['acc_mean']:>7.1f}+/-{r['acc_std']:<5.1f} "
                  f"{r['rob_mean']:>7.3f}+/-{r['rob_std']:<5.3f}")
