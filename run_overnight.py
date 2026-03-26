"""
Overnight experiment suite: Crosstalk scaling + Braid/Diamond benchmarks.

Track A: Crosstalk scaling (butterfly + SCF at multiple N and epsilon)
Track B1: Braid topology full benchmark + crosstalk
Track B2: Diamond topology full benchmark

Results saved incrementally to results_overnight.json.
"""

import json
import math
import os
import time
import torch
import torch.nn as nn
import numpy as np

from physics import PhotonicNeuralNetwork, get_topology_info
from prepare import prepare_data
from evaluate import train_model, evaluate_model
from experiment_crosstalk import PhotonicNeuralNetworkCrosstalk

RESULTS_FILE = "results_overnight.json"
SEEDS = [None, 1, 2, 3]  # canonical 4 seeds


def save_result(result):
    """Append one result to the JSON results file (incremental save)."""
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results.append(result)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_standard(topology, N, dataset, seed, loss_dB=0.2, crossing_loss_dB=0.02):
    """Run one standard training + evaluation."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.seed()
        np.random.seed()

    device = torch.device("cpu")
    train_loader, test_loader, data_info = prepare_data(dataset, pca_dim=N)

    model = PhotonicNeuralNetwork(
        N=N, topology=topology, n_classes=data_info["n_classes"],
        n_photonic_layers=1, loss_per_mzi_dB=loss_dB,
        classifier_hidden=(64,), nonlinearity="photodetect",
        crossing_loss_dB=crossing_loss_dB,
    ).to(device)

    train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
    results = evaluate_model(model, test_loader, device)
    return results["clean_accuracy"] * 100, results["robustness"], train_time


def run_crosstalk(topology, N, dataset, seed, eps, loss_dB=0.2, crossing_loss_dB=0.02):
    """Run one crosstalk training + evaluation."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.seed()
        np.random.seed()

    device = torch.device("cpu")
    train_loader, test_loader, data_info = prepare_data(dataset, pca_dim=N)

    model = PhotonicNeuralNetworkCrosstalk(
        N=N, topology=topology, n_classes=data_info["n_classes"],
        n_photonic_layers=1, loss_per_mzi_dB=loss_dB,
        classifier_hidden=(64,), nonlinearity="photodetect",
        crossing_loss_dB=crossing_loss_dB, crosstalk_eps=eps,
    ).to(device)
    model.N = N
    model.topology_name = topology

    train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
    results = evaluate_model(model, test_loader, device)
    return results["clean_accuracy"] * 100, results["robustness"], train_time


def run_track_a():
    """Track A: Crosstalk scaling study."""
    print("=" * 70)
    print("TRACK A: Crosstalk Scaling Study")
    print("=" * 70)

    configs = [
        ("butterfly", [32, 64, 128]),
        ("scf_fractal", [16, 32, 64]),
    ]
    eps_levels = [0.0, 0.0001, 0.001]
    dataset = "vowel"
    track_results = []

    for topo, sizes in configs:
        for N in sizes:
            for eps in eps_levels:
                accs, robs = [], []
                for seed in SEEDS:
                    t0 = time.time()
                    acc, rob, _ = run_crosstalk(topo, N, dataset, seed, eps)
                    elapsed = time.time() - t0
                    accs.append(acc)
                    robs.append(rob)
                    stag = f"seed={seed}" if seed is not None else "seed=default"
                    print(f"  [Track A] {topo} N={N}, eps={eps}, {stag}: "
                          f"acc={acc:.1f}%, rob={rob:.3f} ({elapsed:.1f}s)")

                result = {
                    "track": "A",
                    "topology": topo,
                    "N": N,
                    "epsilon": eps,
                    "dataset": dataset,
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs, ddof=0)),
                    "rob_mean": float(np.mean(robs)),
                    "rob_std": float(np.std(robs, ddof=0)),
                    "accs": accs,
                    "robs": robs,
                }
                track_results.append(result)
                save_result(result)
                print(f"  >> {topo} N={N} eps={eps}: "
                      f"{result['acc_mean']:.1f}+/-{result['acc_std']:.1f}%, "
                      f"rob={result['rob_mean']:.3f}+/-{result['rob_std']:.3f}")
                print()

    return track_results


def run_track_b1():
    """Track B1: Braid topology full benchmark + crosstalk."""
    print("=" * 70)
    print("TRACK B1: Braid Topology Benchmark")
    print("=" * 70)

    dataset = "vowel"
    track_results = []

    # Standard benchmark at all sizes
    for N in [4, 8, 16, 32, 64, 128]:
        accs, robs = [], []
        for seed in SEEDS:
            t0 = time.time()
            acc, rob, _ = run_standard("braid", N, dataset, seed)
            elapsed = time.time() - t0
            accs.append(acc)
            robs.append(rob)
            stag = f"seed={seed}" if seed is not None else "seed=default"
            print(f"  [Track B1] braid N={N}, {stag}: "
                  f"acc={acc:.1f}%, rob={rob:.3f} ({elapsed:.1f}s)")

        result = {
            "track": "B1",
            "topology": "braid",
            "N": N,
            "epsilon": None,
            "dataset": dataset,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=0)),
            "rob_mean": float(np.mean(robs)),
            "rob_std": float(np.std(robs, ddof=0)),
            "accs": accs,
            "robs": robs,
        }
        track_results.append(result)
        save_result(result)
        print(f"  >> braid N={N}: {result['acc_mean']:.1f}+/-{result['acc_std']:.1f}%, "
              f"rob={result['rob_mean']:.3f}+/-{result['rob_std']:.3f}")
        print()

    # Crosstalk runs at N=16 and N=64
    eps_levels = [0.0, 0.0001, 0.001]
    for N in [16, 64]:
        for eps in eps_levels:
            accs, robs = [], []
            for seed in SEEDS:
                t0 = time.time()
                acc, rob, _ = run_crosstalk("braid", N, dataset, seed, eps)
                elapsed = time.time() - t0
                accs.append(acc)
                robs.append(rob)
                stag = f"seed={seed}" if seed is not None else "seed=default"
                print(f"  [Track B1-XT] braid N={N}, eps={eps}, {stag}: "
                      f"acc={acc:.1f}%, rob={rob:.3f} ({elapsed:.1f}s)")

            result = {
                "track": "B1-crosstalk",
                "topology": "braid",
                "N": N,
                "epsilon": eps,
                "dataset": dataset,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=0)),
                "rob_mean": float(np.mean(robs)),
                "rob_std": float(np.std(robs, ddof=0)),
                "accs": accs,
                "robs": robs,
            }
            track_results.append(result)
            save_result(result)
            print(f"  >> braid N={N} eps={eps}: "
                  f"{result['acc_mean']:.1f}+/-{result['acc_std']:.1f}%, "
                  f"rob={result['rob_mean']:.3f}+/-{result['rob_std']:.3f}")
            print()

    return track_results


def run_track_b2():
    """Track B2: Diamond topology full benchmark."""
    print("=" * 70)
    print("TRACK B2: Diamond Topology Benchmark")
    print("=" * 70)

    dataset = "vowel"
    track_results = []

    for N in [4, 8, 16, 32, 64]:
        accs, robs = [], []
        for seed in SEEDS:
            t0 = time.time()
            acc, rob, _ = run_standard("diamond", N, dataset, seed)
            elapsed = time.time() - t0
            accs.append(acc)
            robs.append(rob)
            stag = f"seed={seed}" if seed is not None else "seed=default"
            print(f"  [Track B2] diamond N={N}, {stag}: "
                  f"acc={acc:.1f}%, rob={rob:.3f} ({elapsed:.1f}s)")

        result = {
            "track": "B2",
            "topology": "diamond",
            "N": N,
            "epsilon": None,
            "dataset": dataset,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=0)),
            "rob_mean": float(np.mean(robs)),
            "rob_std": float(np.std(robs, ddof=0)),
            "accs": accs,
            "robs": robs,
        }
        track_results.append(result)
        save_result(result)
        print(f"  >> diamond N={N}: {result['acc_mean']:.1f}+/-{result['acc_std']:.1f}%, "
              f"rob={result['rob_mean']:.3f}+/-{result['rob_std']:.3f}")
        print()

    return track_results


def print_summary(all_results):
    """Print final summary tables."""
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Track A: Crosstalk tables
    for topo in ["butterfly", "scf_fractal"]:
        print(f"\n{topo.upper()} Crosstalk Scaling:")
        print(f"{'N':>5} | {'eps=0':>15} | {'eps=0.0001':>15} | {'eps=0.001':>15} | {'Shift':>8}")
        print("-" * 70)
        track_a = [r for r in all_results if r["track"] == "A" and r["topology"] == topo]
        sizes = sorted(set(r["N"] for r in track_a))
        for N in sizes:
            row = {}
            for r in track_a:
                if r["N"] == N:
                    row[r["epsilon"]] = r
            baseline = row.get(0.0, {}).get("acc_mean", 0)
            worst = row.get(0.001, {}).get("acc_mean", 0)
            shift = worst - baseline if baseline and worst else 0
            for eps in [0.0, 0.0001, 0.001]:
                r = row.get(eps, {})
                val = f"{r.get('acc_mean', 0):.1f}+/-{r.get('acc_std', 0):.1f}%"
                row[f"str_{eps}"] = val
            print(f"{N:>5} | {row.get('str_0.0', ''):>15} | "
                  f"{row.get('str_0.0001', ''):>15} | "
                  f"{row.get('str_0.001', ''):>15} | {shift:>+.1f}pp")

    # Track B1: Braid
    print(f"\nBRAID Standard Benchmark:")
    print(f"{'N':>5} | {'Acc (%)':>15} | {'Rob':>15} | {'MZIs':>6} | {'Depth':>6} | {'Cross':>6}")
    print("-" * 70)
    track_b1 = [r for r in all_results
                if r["track"] == "B1" and r["topology"] == "braid"]
    for r in sorted(track_b1, key=lambda x: x["N"]):
        N = r["N"]
        _, n_mzis, depth = get_topology_info("braid", N)
        # Get crossing count
        from physics import braid_topology
        layers = braid_topology(N)
        cx = 0
        for layer in layers:
            for (pi, pj) in layer:
                span = abs(pj - pi)
                if span > 1:
                    cx += 2 * (span - 1) + (span - 1)
        print(f"{N:>5} | {r['acc_mean']:>7.1f}+/-{r['acc_std']:<5.1f} | "
              f"{r['rob_mean']:>7.3f}+/-{r['rob_std']:<5.3f} | "
              f"{n_mzis:>6} | {depth:>6} | {cx:>6}")

    # Track B2: Diamond
    print(f"\nDIAMOND Standard Benchmark:")
    print(f"{'N':>5} | {'Acc (%)':>15} | {'Rob':>15} | {'MZIs':>6} | {'Depth':>6}")
    print("-" * 65)
    track_b2 = [r for r in all_results
                if r["track"] == "B2" and r["topology"] == "diamond"]
    for r in sorted(track_b2, key=lambda x: x["N"]):
        N = r["N"]
        _, n_mzis, depth = get_topology_info("diamond", N)
        print(f"{N:>5} | {r['acc_mean']:>7.1f}+/-{r['acc_std']:<5.1f} | "
              f"{r['rob_mean']:>7.3f}+/-{r['rob_std']:<5.3f} | "
              f"{n_mzis:>6} | {depth:>6}")

    # Braid crosstalk
    print(f"\nBRAID Crosstalk:")
    print(f"{'N':>5} | {'eps=0':>15} | {'eps=0.0001':>15} | {'eps=0.001':>15} | {'Shift':>8}")
    print("-" * 70)
    track_b1_xt = [r for r in all_results if r["track"] == "B1-crosstalk"]
    for N in sorted(set(r["N"] for r in track_b1_xt)):
        row = {}
        for r in track_b1_xt:
            if r["N"] == N:
                row[r["epsilon"]] = r
        baseline = row.get(0.0, {}).get("acc_mean", 0)
        worst = row.get(0.001, {}).get("acc_mean", 0)
        shift = worst - baseline if baseline and worst else 0
        for eps in [0.0, 0.0001, 0.001]:
            r = row.get(eps, {})
            row[f"str_{eps}"] = f"{r.get('acc_mean', 0):.1f}+/-{r.get('acc_std', 0):.1f}%"
        print(f"{N:>5} | {row.get('str_0.0', ''):>15} | "
              f"{row.get('str_0.0001', ''):>15} | "
              f"{row.get('str_0.001', ''):>15} | {shift:>+.1f}pp")


if __name__ == "__main__":
    # Clear previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    t_start = time.time()
    all_results = []

    # Priority order
    all_results.extend(run_track_a())
    all_results.extend(run_track_b1())
    all_results.extend(run_track_b2())

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")

    print_summary(all_results)
