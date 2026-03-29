"""
Task 4a: Benchmark L-BFGS vs Adam for mesh optimization at N=32.
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import PhotonicMesh
from sdm_experiment.sdm_channel import generate_channel
from sdm_experiment.sdm_optimize import get_mesh_matrix


def optimize_adam(topology, N, H, mzi_loss_dB, crossing_loss_dB, mesh_seed):
    """Adam optimization (current approach): 2000 steps, lr=0.01, milestones."""
    torch.manual_seed(mesh_seed)
    mesh = PhotonicMesh(N, topology=topology, loss_per_mzi_dB=mzi_loss_dB,
                        crossing_loss_dB=crossing_loss_dB)
    H_t = H.clone().detach()
    I = torch.eye(N, dtype=torch.complex64)

    optimizer = torch.optim.Adam(mesh.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    t0 = time.time()
    loss_history = []
    for step in range(2000):
        optimizer.zero_grad()
        M = get_mesh_matrix(mesh)
        loss = torch.norm(M @ H_t - I) ** 2
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

    wall_time = time.time() - t0

    with torch.no_grad():
        M = get_mesh_matrix(mesh)
        E = M @ H_t
        fidelity = 1.0 - torch.norm(E - I).item() ** 2 / (2 * N)

    return {
        "fidelity": fidelity,
        "final_loss": loss_history[-1],
        "wall_time_sec": round(wall_time, 2),
        "steps": 2000,
        "loss_history": loss_history,
    }


def optimize_lbfgs(topology, N, H, mzi_loss_dB, crossing_loss_dB, mesh_seed,
                   lr=1.0, max_outer=200, max_iter=20, history_size=10):
    """L-BFGS optimization with closure."""
    torch.manual_seed(mesh_seed)
    mesh = PhotonicMesh(N, topology=topology, loss_per_mzi_dB=mzi_loss_dB,
                        crossing_loss_dB=crossing_loss_dB)
    H_t = H.clone().detach()
    I = torch.eye(N, dtype=torch.complex64)

    optimizer = torch.optim.LBFGS(mesh.parameters(), lr=lr,
                                   max_iter=max_iter, history_size=history_size,
                                   line_search_fn='strong_wolfe')

    t0 = time.time()
    loss_history = []
    n_evals = [0]

    for outer in range(max_outer):
        def closure():
            optimizer.zero_grad()
            M = get_mesh_matrix(mesh)
            loss = torch.norm(M @ H_t - I) ** 2
            loss.backward()
            n_evals[0] += 1
            return loss

        loss = optimizer.step(closure)

        if loss is not None:
            loss_val = loss.item()
        else:
            # Fallback: compute loss
            with torch.no_grad():
                M = get_mesh_matrix(mesh)
                loss_val = torch.norm(M @ H_t - I).item() ** 2

        loss_history.append(loss_val)

        # Check for NaN
        if np.isnan(loss_val):
            break

        # Early stopping if converged
        if len(loss_history) >= 10:
            if np.std(loss_history[-10:]) < 1e-6:
                break

    wall_time = time.time() - t0

    with torch.no_grad():
        M = get_mesh_matrix(mesh)
        E = M @ H_t
        fidelity = 1.0 - torch.norm(E - I).item() ** 2 / (2 * N)

    return {
        "fidelity": fidelity,
        "final_loss": loss_history[-1] if loss_history else float('nan'),
        "wall_time_sec": round(wall_time, 2),
        "steps": len(loss_history),
        "evals": n_evals[0],
        "loss_history": loss_history,
    }


def run_benchmark():
    N = 32
    mesh_seed = 2000

    test_cases = [
        ("clements", 0.5),
        ("clements", 5.0),
        ("butterfly", 0.5),
        ("butterfly", 5.0),
        ("reck", 5.0),
    ]

    loss_configs = [
        ("lossy", 0.2, 0.02),
        ("lossless", 0.0, 0.0),
    ]

    print("=" * 90)
    print(f"Task 4a: L-BFGS vs Adam benchmark (N={N})")
    print("=" * 90)

    results = []

    for topo, sigma in test_cases:
        H = generate_channel(N, sigma, seed=1000)

        for loss_label, mzi_loss, cross_loss in loss_configs:
            print(f"\n--- {topo} σ={sigma} ({loss_label}) ---")

            # Adam
            r_adam = optimize_adam(topo, N, H, mzi_loss, cross_loss, mesh_seed)
            print(f"  Adam:  fid={r_adam['fidelity']:.6f}, "
                  f"loss={r_adam['final_loss']:.6f}, "
                  f"time={r_adam['wall_time_sec']:.1f}s, "
                  f"steps={r_adam['steps']}")

            # L-BFGS (try lr=1.0 first)
            r_lbfgs = optimize_lbfgs(topo, N, H, mzi_loss, cross_loss, mesh_seed, lr=1.0)
            if np.isnan(r_lbfgs['final_loss']):
                print("  L-BFGS lr=1.0: NaN! Retrying lr=0.1...")
                r_lbfgs = optimize_lbfgs(topo, N, H, mzi_loss, cross_loss, mesh_seed, lr=0.1)

            print(f"  L-BFGS: fid={r_lbfgs['fidelity']:.6f}, "
                  f"loss={r_lbfgs['final_loss']:.6f}, "
                  f"time={r_lbfgs['wall_time_sec']:.1f}s, "
                  f"steps={r_lbfgs['steps']}, evals={r_lbfgs['evals']}")

            fid_diff = r_lbfgs['fidelity'] - r_adam['fidelity']
            speedup = r_adam['wall_time_sec'] / r_lbfgs['wall_time_sec'] if r_lbfgs['wall_time_sec'] > 0 else 0
            print(f"  Delta fid: {fid_diff:+.6f}, speedup: {speedup:.2f}x")

            results.append({
                "topology": topo,
                "sigma": sigma,
                "loss_config": loss_label,
                "adam_fidelity": r_adam['fidelity'],
                "adam_time": r_adam['wall_time_sec'],
                "lbfgs_fidelity": r_lbfgs['fidelity'],
                "lbfgs_time": r_lbfgs['wall_time_sec'],
                "lbfgs_steps": r_lbfgs['steps'],
                "lbfgs_evals": r_lbfgs['evals'],
                "fid_diff": fid_diff,
                "speedup": speedup,
            })

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Case':<30} {'Config':<10} {'Adam fid':>10} {'L-BFGS fid':>11} "
          f"{'Δfid':>8} {'Adam(s)':>8} {'LBFGS(s)':>9} {'Speedup':>8}")
    print("-" * 90)
    for r in results:
        case = f"{r['topology']} σ={r['sigma']}"
        print(f"{case:<30} {r['loss_config']:<10} {r['adam_fidelity']:>10.6f} "
              f"{r['lbfgs_fidelity']:>11.6f} {r['fid_diff']:>+8.5f} "
              f"{r['adam_time']:>8.1f} {r['lbfgs_time']:>9.1f} "
              f"{r['speedup']:>7.2f}x")

    # Verdict
    lbfgs_wins = sum(1 for r in results
                     if r['speedup'] > 1.0 and r['fid_diff'] > -0.005)
    total = len(results)
    print(f"\nL-BFGS wins (faster AND fidelity within 0.005): {lbfgs_wins}/{total}")

    return results


if __name__ == "__main__":
    run_benchmark()
