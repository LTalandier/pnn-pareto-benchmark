"""
SDM channel equalization via photonic mesh optimization.

For each (topology, channel H, N):
  1. Initialize mesh phases randomly
  2. Extract M_mesh as the optical transfer matrix
  3. Minimize ||M_mesh @ H - I||_F^2
  4. Report fidelity and SNR
"""

import sys
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import PhotonicMesh
from sdm_experiment.sdm_channel import generate_channel


def get_mesh_matrix(mesh):
    """
    Extract the N x N optical transfer matrix from a PhotonicMesh.

    The mesh uses right-multiply convention: output = input @ M.
    We return M transposed to standard left-multiply convention: output = M @ input.
    This way M_mesh @ H gives the composed transfer matrix.
    """
    mesh._cached_matrix = None  # force recompute
    mesh.train()  # ensure we go through differentiable path

    # Build matrix by multiplying layer matrices
    M = None
    for layer_idx in range(mesh._n_layers):
        L = mesh._build_layer_matrix(layer_idx, noise_sigma=0.0)
        if M is None:
            M = L
        else:
            M = M @ L
    # M is in right-multiply convention (state @ M).
    # Transpose to left-multiply convention (M @ state).
    return M.T


def _compute_metrics(mesh, H_t, N):
    """Compute fidelity and SNR metrics."""
    M_mesh = get_mesh_matrix(mesh)
    E = M_mesh @ H_t
    I = torch.eye(N, dtype=torch.complex64)
    fidelity = 1.0 - torch.norm(E - I).item() ** 2 / (2 * N)
    E_abs2 = torch.abs(E) ** 2
    diag_power = torch.diag(E_abs2)
    off_diag_power = E_abs2.sum(dim=1) - diag_power
    snr_per_ch = diag_power / (off_diag_power + 1e-10)
    snr_db = 10 * torch.log10(snr_per_ch).mean().item()
    return fidelity, snr_db


def _optimize_lbfgs(topology, N, H, mzi_loss_dB, crossing_loss_dB,
                    seed, max_steps=200, lr=1.0, verbose=False):
    """Single optimization run using L-BFGS. Returns result dict."""
    torch.manual_seed(seed)

    mesh = PhotonicMesh(N, topology=topology, loss_per_mzi_dB=mzi_loss_dB,
                        crossing_loss_dB=crossing_loss_dB)

    H_t = H.clone().detach()
    I = torch.eye(N, dtype=torch.complex64)

    optimizer = torch.optim.LBFGS(mesh.parameters(), lr=lr,
                                   max_iter=20, history_size=10,
                                   line_search_fn='strong_wolfe')

    loss_history = []
    t0 = time.time()
    converged = False

    for step in range(max_steps):
        def closure():
            optimizer.zero_grad()
            M_mesh = get_mesh_matrix(mesh)
            E = M_mesh @ H_t
            loss = torch.norm(E - I) ** 2
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_val = loss.item()

        if math.isnan(loss_val):
            # NaN — restart with lower lr
            if verbose:
                print(f"  NaN at step {step}, restarting with lr=0.1")
            torch.manual_seed(seed + 500)
            mesh = PhotonicMesh(N, topology=topology, loss_per_mzi_dB=mzi_loss_dB,
                                crossing_loss_dB=crossing_loss_dB)
            optimizer = torch.optim.LBFGS(mesh.parameters(), lr=0.1,
                                           max_iter=20, history_size=10,
                                           line_search_fn='strong_wolfe')
            loss_history = []
            continue

        loss_history.append(loss_val)

        if verbose and step % 50 == 0:
            print(f"  step {step:5d}: loss={loss_val:.6f}")

        # Early stop if converged
        if len(loss_history) >= 10:
            recent = loss_history[-10:]
            if max(recent) - min(recent) < 1e-8:
                converged = True
                break

    if not converged and len(loss_history) >= 10:
        recent = loss_history[-10:]
        converged = max(recent) - min(recent) < 1e-6

    wall_time = time.time() - t0

    with torch.no_grad():
        fidelity, snr_db = _compute_metrics(mesh, H_t, N)

    return {
        "fidelity": fidelity,
        "snr_db": snr_db,
        "converged": converged,
        "steps": len(loss_history),
        "wall_time_sec": round(wall_time, 2),
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else float('inf'),
    }


def _optimize_adam(topology, N, H, lr, max_steps, mzi_loss_dB,
                   crossing_loss_dB, seed, verbose):
    """Single optimization run using Adam. Returns result dict."""
    torch.manual_seed(seed)

    mesh = PhotonicMesh(N, topology=topology, loss_per_mzi_dB=mzi_loss_dB,
                        crossing_loss_dB=crossing_loss_dB)

    H_t = H.clone().detach()
    I = torch.eye(N, dtype=torch.complex64)

    optimizer = torch.optim.Adam(mesh.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    loss_history = []
    t0 = time.time()

    for step in range(max_steps):
        optimizer.zero_grad()
        M_mesh = get_mesh_matrix(mesh)
        E = M_mesh @ H_t
        loss = torch.norm(E - I) ** 2
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and step % 500 == 0:
            print(f"  step {step:5d}: loss={loss_val:.6f}")

    # Convergence detection
    def _is_plateaued(history, window=200):
        if len(history) < window:
            return False
        return float(np.std(history[-window:])) < 1e-4

    converged = _is_plateaued(loss_history)

    wall_time = time.time() - t0

    with torch.no_grad():
        fidelity, snr_db = _compute_metrics(mesh, H_t, N)

    return {
        "fidelity": fidelity,
        "snr_db": snr_db,
        "converged": converged,
        "steps": len(loss_history),
        "wall_time_sec": round(wall_time, 2),
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
    }


def optimize_equalization(topology, N, H, lr=0.01, max_steps=2000,
                          mzi_loss_dB=0.2, crossing_loss_dB=0.02,
                          mesh_seed=None, n_restarts=1, verbose=False,
                          optimizer='lbfgs'):
    """
    Optimize a photonic mesh to implement H^{-1}.

    Args:
        topology: topology name string
        N: mesh size
        H: [N, N] complex channel matrix
        lr: initial learning rate (0.01 for Adam, 1.0 for L-BFGS)
        max_steps: step budget per restart (2000 for Adam, 200 for L-BFGS)
        mzi_loss_dB: insertion loss per MZI
        crossing_loss_dB: crossing loss per crossing
        mesh_seed: base seed for mesh initialization (each restart offsets by +1000)
        n_restarts: number of random restarts (default 1)
        verbose: print progress
        optimizer: 'lbfgs' or 'adam'

    Returns:
        dict with fidelity, snr_db, converged, steps, wall_time_sec, loss_history
    """
    base_seed = mesh_seed if mesh_seed is not None else 0
    best_result = None

    for restart in range(n_restarts):
        seed = base_seed + restart * 1000
        if verbose and n_restarts > 1:
            print(f"  [restart {restart+1}/{n_restarts}, seed={seed}]")

        if optimizer == 'lbfgs':
            lbfgs_lr = lr if lr != 0.01 else 1.0  # default lr for lbfgs
            lbfgs_steps = max_steps if max_steps != 2000 else 200
            result = _optimize_lbfgs(
                topology, N, H, mzi_loss_dB, crossing_loss_dB,
                seed, max_steps=lbfgs_steps, lr=lbfgs_lr, verbose=verbose)
        else:
            result = _optimize_adam(
                topology, N, H, lr, max_steps, mzi_loss_dB,
                crossing_loss_dB, seed, verbose)

        if best_result is None or result['final_loss'] < best_result['final_loss']:
            best_result = result

    return best_result


def run_task2_validation():
    """
    Task 2: Single-point optimization test.
    N=8, butterfly + clements, sigma = 0.01, 0.5, 5.0, 3 channels each.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = 8
    topologies = ['butterfly', 'clements']
    sigmas = [0.01, 0.5, 5.0]
    n_channels = 3

    all_results = []

    print("=" * 70)
    print(f"Task 2: Single-point optimization test (N={N})")
    print("=" * 70)

    for topo in topologies:
        for sigma in sigmas:
            for ch_idx in range(n_channels):
                ch_seed = 1000 + ch_idx
                mesh_seed = 2000 + ch_idx
                H = generate_channel(N, sigma, seed=ch_seed)

                print(f"\n{topo} | sigma={sigma} | channel={ch_idx}")
                result = optimize_equalization(
                    topology=topo, N=N, H=H,
                    mesh_seed=mesh_seed, verbose=True)

                result["topology"] = topo
                result["sigma"] = sigma
                result["channel_seed"] = ch_seed
                all_results.append(result)

                print(f"  -> fidelity={result['fidelity']:.6f}, "
                      f"SNR={result['snr_db']:.1f} dB, "
                      f"time={result['wall_time_sec']:.1f}s, "
                      f"converged={result['converged']}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Topology':<12} {'sigma':<8} {'ch':<4} {'Fidelity':<10} "
          f"{'SNR(dB)':<10} {'Time(s)':<10} {'Conv':<6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['topology']:<12} {r['sigma']:<8} {r['channel_seed']:<4} "
              f"{r['fidelity']:<10.6f} {r['snr_db']:<10.1f} "
              f"{r['wall_time_sec']:<10.1f} {r['converged']}")

    # Plot convergence curves
    fig, axes = plt.subplots(len(sigmas), len(topologies),
                             figsize=(12, 3 * len(sigmas)), squeeze=False)

    for col, topo in enumerate(topologies):
        for row, sigma in enumerate(sigmas):
            ax = axes[row][col]
            for r in all_results:
                if r['topology'] == topo and r['sigma'] == sigma:
                    ax.semilogy(r['loss_history'], alpha=0.7,
                                label=f"ch {r['channel_seed']}")
            ax.set_title(f'{topo}, σ={sigma}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Convergence curves (N={N})', fontsize=14)
    plt.tight_layout()

    os.makedirs('results/figures', exist_ok=True)
    fig_path = 'results/figures/sdm_task2_convergence.pdf'
    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved convergence plot: {fig_path}")

    return all_results


if __name__ == "__main__":
    results = run_task2_validation()
