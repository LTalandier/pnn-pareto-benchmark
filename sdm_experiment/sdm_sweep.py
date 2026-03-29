"""
SDM full parameter sweep with parallel execution.

Sweeps: N x topology x sigma x channel_seed, with lossy and lossless runs.
Results saved to results/sdm_sweep.json.
"""

import sys
import os
import json
import time
import multiprocessing as mp
from functools import partial

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdm_experiment.sdm_channel import generate_channel
from sdm_experiment.sdm_optimize import optimize_equalization


TOPOLOGIES = ['clements', 'reck', 'butterfly', 'braid', 'diamond', 'scf_fractal']
SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def run_single_job(job):
    """
    Run lossy + lossless optimization for a single (topology, N, sigma, channel_seed).
    Called in a worker process.
    """
    torch.set_num_threads(1)

    topology = job['topology']
    N = job['N']
    sigma = job['sigma']
    ch_seed = job['channel_seed']
    mesh_seed = job['mesh_seed']
    n_restarts = job.get('n_restarts', 1)
    opt = job.get('optimizer', 'lbfgs')

    H = generate_channel(N, sigma, seed=ch_seed)

    # Lossy run (primary)
    r_lossy = optimize_equalization(
        topology=topology, N=N, H=H,
        mzi_loss_dB=0.2, crossing_loss_dB=0.02,
        mesh_seed=mesh_seed, n_restarts=n_restarts,
        verbose=False, optimizer=opt)

    # Lossless run (expressivity baseline)
    r_lossless = optimize_equalization(
        topology=topology, N=N, H=H,
        mzi_loss_dB=0.0, crossing_loss_dB=0.0,
        mesh_seed=mesh_seed, n_restarts=n_restarts,
        verbose=False, optimizer=opt)

    return {
        "topology": topology,
        "N": N,
        "sigma": sigma,
        "channel_seed": ch_seed,
        "fidelity_raw": round(r_lossy['fidelity'], 6),
        "snr_db_raw": round(r_lossy['snr_db'], 2),
        "converged_raw": r_lossy['converged'],
        "steps_raw": r_lossy['steps'],
        "fidelity_lossless": round(r_lossless['fidelity'], 6),
        "snr_db_lossless": round(r_lossless['snr_db'], 2),
        "converged_lossless": r_lossless['converged'],
        "steps_lossless": r_lossless['steps'],
        "wall_time_sec": round(r_lossy['wall_time_sec'] + r_lossless['wall_time_sec'], 2),
    }


def build_job_list(N_values, n_channels=20, topologies=None,
                   n_restarts=1, optimizer='lbfgs'):
    """Build list of all jobs for the sweep."""
    if topologies is None:
        topologies = TOPOLOGIES
    jobs = []
    for N in N_values:
        for topo in topologies:
            for sigma in SIGMAS:
                for ch_idx in range(n_channels):
                    jobs.append({
                        'topology': topo,
                        'N': N,
                        'sigma': sigma,
                        'channel_seed': 1000 + ch_idx,
                        'mesh_seed': 2000 + ch_idx,
                        'n_restarts': n_restarts,
                        'optimizer': optimizer,
                    })
    return jobs


def run_sweep(N_values, n_channels=20, n_workers=16, output_path='results/sdm_sweep.json',
              topologies=None, n_restarts=1, optimizer='lbfgs'):
    """Run the full sweep with multiprocessing."""
    jobs = build_job_list(N_values, n_channels, topologies=topologies,
                          n_restarts=n_restarts, optimizer=optimizer)
    topos_used = topologies or TOPOLOGIES
    total = len(jobs)

    print(f"SDM Sweep: {total} jobs ({total * 2} optimizations: lossy + lossless)")
    print(f"  N values: {N_values}")
    print(f"  Topologies: {topos_used}")
    print(f"  Sigmas: {SIGMAS}")
    print(f"  Optimizer: {optimizer}, Restarts: {n_restarts}")
    print(f"  Channels per config: {n_channels}")
    print(f"  Workers: {n_workers}")
    print()

    # Load existing results if appending
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        # Build set of already-completed keys
        done_keys = set()
        for r in existing_results:
            key = (r['topology'], r['N'], r['sigma'], r['channel_seed'])
            done_keys.add(key)
        # Filter out already-done jobs
        jobs = [j for j in jobs
                if (j['topology'], j['N'], j['sigma'], j['channel_seed']) not in done_keys]
        if len(jobs) < total:
            print(f"  Resuming: {total - len(jobs)} already done, {len(jobs)} remaining")
            total_new = len(jobs)
        else:
            total_new = total
    else:
        total_new = total

    if total_new == 0:
        print("All jobs already completed!")
        return existing_results

    t0 = time.time()
    results = list(existing_results)
    completed = 0

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_job, jobs):
            results.append(result)
            completed += 1

            if completed % 50 == 0 or completed == total_new:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_new - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total_new}] "
                      f"{elapsed:.0f}s elapsed, {rate:.1f} jobs/s, "
                      f"ETA {eta:.0f}s | "
                      f"last: {result['topology']} N={result['N']} "
                      f"σ={result['sigma']} "
                      f"fid_raw={result['fidelity_raw']:.4f} "
                      f"fid_ll={result['fidelity_lossless']:.4f}")

            # Checkpoint every 200 jobs
            if completed % 200 == 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)

    total_time = time.time() - t0

    # Final save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {total_new} jobs in {total_time:.0f}s ({total_new/total_time:.1f} jobs/s)")
    print(f"Saved to {output_path}")

    return results


def print_summary(results, N_values=None):
    """Print summary statistics table."""
    if N_values is None:
        N_values = sorted(set(r['N'] for r in results))

    print("\n" + "=" * 100)
    print("SUMMARY: Mean fidelity (raw / lossless) ± std")
    print("=" * 100)

    for N in N_values:
        print(f"\n--- N = {N} ---")
        header = f"{'Topology':<14}"
        for sigma in SIGMAS:
            header += f" {'σ='+str(sigma):>12}"
        print(header)
        print("-" * (14 + 13 * len(SIGMAS)))

        for topo in TOPOLOGIES:
            row_raw = f"{topo:<14}"
            row_ll = f"{'(lossless)':<14}"
            for sigma in SIGMAS:
                subset = [r for r in results
                          if r['topology'] == topo and r['N'] == N and r['sigma'] == sigma]
                if subset:
                    fids_raw = [r['fidelity_raw'] for r in subset]
                    fids_ll = [r['fidelity_lossless'] for r in subset]
                    row_raw += f" {np.mean(fids_raw):.3f}±{np.std(fids_raw):.3f}"
                    row_ll += f" {np.mean(fids_ll):.3f}±{np.std(fids_ll):.3f}"
                else:
                    row_raw += f" {'---':>12}"
                    row_ll += f" {'---':>12}"
            print(row_raw)
            print(row_ll)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', nargs='+', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--channels', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', default='results/sdm_sweep.json')
    parser.add_argument('--topologies', nargs='+', default=None)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--optimizer', default='lbfgs', choices=['lbfgs', 'adam'])
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    if args.summary_only:
        with open(args.output, 'r') as f:
            results = json.load(f)
        print_summary(results, args.N)
    else:
        results = run_sweep(args.N, args.channels, args.workers, args.output,
                            topologies=args.topologies, n_restarts=args.restarts,
                            optimizer=args.optimizer)
        print_summary(results, args.N)
