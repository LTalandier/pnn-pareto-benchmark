"""
Seed sweep for N=4 and N=8 vowel (N=16 already done).
4 topologies × 2 sizes × 3 seeds = 24 experiments. CPU.
"""
import subprocess
import sys
import os
import time
import textwrap

os.chdir("/root/pnn-pareto")

TOPOLOGIES = ["clements", "reck", "butterfly", "scf_fractal"]
MESH_SIZES = [4, 8]
SEEDS = [1, 2, 3]

def write_train_py(topology, mesh_size, seed):
    code = textwrap.dedent(f'''\
        import torch
        import numpy as np
        from physics import PhotonicNeuralNetwork, TOPOLOGIES
        from prepare import prepare_data
        from evaluate import (evaluate_model, train_model, log_experiment,
                              get_experiment_count)

        def run_experiment():
            torch.manual_seed({seed})
            np.random.seed({seed})

            device = torch.device("cpu")
            train_loader, test_loader, data_info = prepare_data("vowel", pca_dim={mesh_size})
            n_classes = data_info["n_classes"]

            model = PhotonicNeuralNetwork(
                N={mesh_size}, topology="{topology}", n_classes=n_classes,
                n_photonic_layers=1, loss_per_mzi_dB=0.2,
                classifier_hidden=(64,), nonlinearity="photodetect",
            ).to(device)
            model.N = {mesh_size}
            model.topology_name = "{topology}"

            print(f"{{model.get_mzi_count()}} MZIs, depth {{model.get_optical_depth()}}")

            train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
            print(f"Training: {{train_time:.1f}}s")

            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time

            config = dict(
                topology="{topology}", mesh_size={mesh_size}, dataset="vowel",
                n_photonic_layers=1, loss_per_mzi_dB=0.2,
                classifier_hidden=[64], nonlinearity="photodetect",
                n_epochs=100, learning_rate=0.005,
                noise_aware_training=False, train_noise_sigma=0,
                seed={seed},
            )

            desc = f"{topology} N={mesh_size} vowel seed={seed}"
            exp_id = log_experiment(results, description=desc, config=config)
            print(f"acc={{results['clean_accuracy']:.4f}} rob={{results['robustness']:.4f}}")

        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)


total = len(TOPOLOGIES) * len(MESH_SIZES) * len(SEEDS)
failed = []
print(f"=== SEED SWEEP N=4,8: {total} experiments ===")
t_start = time.time()

for mesh_size in MESH_SIZES:
    for topo in TOPOLOGIES:
        for seed in SEEDS:
            tag = f"{topo} N={mesh_size} vowel seed={seed}"
            print(f"\n{tag}")

            write_train_py(topo, mesh_size, seed)

            try:
                result = subprocess.run(
                    [sys.executable, "train.py"],
                    capture_output=True, text=True, timeout=300)
                for line in result.stdout.strip().split("\n"):
                    if "acc=" in line or "MZIs" in line or "Training" in line:
                        print(f"  {line.strip()}")
                if result.returncode != 0:
                    print(f"  FAILED: {result.stderr[-200:]}")
                    failed.append(tag)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT")
                failed.append(tag)

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"DONE in {elapsed:.0f}s — {total - len(failed)}/{total} succeeded")
if failed:
    for f_item in failed:
        print(f"  FAILED: {f_item}")
print(f"{'='*60}")
