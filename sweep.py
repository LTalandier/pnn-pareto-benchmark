"""
Phase 1 Systematic Sweep — runs all 48 experiments.
Writes train.py for each config, runs it, commits.
Skips already-completed experiments found in the log.
"""
import subprocess
import sys
import os
import time
import textwrap
import json

os.chdir("/root/pnn-pareto")

TOPOLOGIES = ["clements", "reck", "butterfly", "scf_fractal"]
MESH_SIZES = [4, 8, 16]
DATASETS = ["vowel", "fmnist"]

def get_completed_experiments():
    """Read log and return set of (topology, mesh_size, dataset) already done."""
    log_file = "results/experiment_log.jsonl"
    completed = set()
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                cfg = entry.get("config", {})
                key = (cfg.get("topology"), cfg.get("mesh_size"), cfg.get("dataset"))
                completed.add(key)
    return completed

def write_train_py(topology, mesh_size, dataset, n_epochs=100):
    code = textwrap.dedent(f'''\
        import torch
        from physics import PhotonicNeuralNetwork, TOPOLOGIES
        from prepare import prepare_data
        from evaluate import (evaluate_model, train_model, log_experiment,
                              get_experiment_count)

        TOPOLOGY = "{topology}"
        MESH_SIZE = {mesh_size}
        DATASET = "{dataset}"
        N_PHOTONIC_LAYERS = 1
        LOSS_PER_MZI_DB = 0.2
        CLASSIFIER_HIDDEN = (64,)
        NONLINEARITY = "photodetect"
        N_EPOCHS = {n_epochs}
        LEARNING_RATE = 0.005

        def run_experiment():
            device = torch.device("cpu")
            print(f"EXPERIMENT: {{TOPOLOGY}} N={{MESH_SIZE}} on {{DATASET}} [{{device}}]")

            train_loader, test_loader, data_info = prepare_data(DATASET, pca_dim=MESH_SIZE)
            n_classes = data_info["n_classes"]

            model = PhotonicNeuralNetwork(
                N=MESH_SIZE, topology=TOPOLOGY, n_classes=n_classes,
                n_photonic_layers=N_PHOTONIC_LAYERS, loss_per_mzi_dB=LOSS_PER_MZI_DB,
                classifier_hidden=CLASSIFIER_HIDDEN, nonlinearity=NONLINEARITY,
            ).to(device)
            model.N = MESH_SIZE
            model.topology_name = TOPOLOGY

            print(f"Model: {{model.get_mzi_count()}} MZIs, depth {{model.get_optical_depth()}}")

            train_time = train_model(model, train_loader, device,
                                     n_epochs=N_EPOCHS, lr=LEARNING_RATE)
            print(f"Training: {{train_time:.1f}}s")

            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time

            config = dict(
                topology=TOPOLOGY, mesh_size=MESH_SIZE, dataset=DATASET,
                n_photonic_layers=N_PHOTONIC_LAYERS, loss_per_mzi_dB=LOSS_PER_MZI_DB,
                classifier_hidden=list(CLASSIFIER_HIDDEN), nonlinearity=NONLINEARITY,
                n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
                noise_aware_training=False, train_noise_sigma=0,
            )

            desc = f"{{TOPOLOGY}} N={{MESH_SIZE}} {{DATASET}} layers=1 loss=0.2dB"
            exp_id = log_experiment(results, description=desc, config=config)
            print(f"RESULT: clean_acc={{results['clean_accuracy']:.4f}} robustness={{results['robustness']:.4f}}")
            return exp_id, results.get("clean_accuracy", 0)

        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)


completed = get_completed_experiments()
total = len(TOPOLOGIES) * len(MESH_SIZES) * len(DATASETS)
skipped = 0
done = 0
failed = []

print(f"=== PHASE 1 SWEEP: {total} experiments ===")
print(f"Already completed: {len(completed)}")
print(f"Topologies: {TOPOLOGIES}")
print(f"Mesh sizes: {MESH_SIZES}")
print(f"Datasets: {DATASETS}")
print()
t_start = time.time()

for dataset in DATASETS:
    for mesh_size in MESH_SIZES:
        for topology in TOPOLOGIES:
            done += 1
            tag = f"{topology} N={mesh_size} {dataset}"

            if (topology, mesh_size, dataset) in completed:
                print(f"\n[{done}/{total}] {tag} — SKIPPED (already done)")
                skipped += 1
                continue

            # Fmnist has 60K samples so needs longer; N=32 also needs more
            if dataset == "fmnist" and mesh_size >= 32:
                timeout = 5400
            elif dataset == "fmnist" or mesh_size >= 32:
                timeout = 3600
            else:
                timeout = 900
            print(f"\n[{done}/{total}] {tag} (timeout={timeout}s)")
            print("-" * 50)

            # Fewer epochs for fmnist (60K samples vs ~550 for vowel)
            if dataset == "fmnist" and mesh_size >= 16:
                n_epochs = 15
            elif dataset == "fmnist":
                n_epochs = 30
            else:
                n_epochs = 100
            write_train_py(topology, mesh_size, dataset, n_epochs=n_epochs)

            try:
                result = subprocess.run(
                    [sys.executable, "train.py"],
                    capture_output=True, text=True, timeout=timeout)
                print(result.stdout[-300:] if result.stdout else "")
                if result.returncode != 0:
                    print(f"  FAILED: {result.stderr[-300:]}")
                    failed.append(tag)
                else:
                    os.system(f'cd /root/pnn-pareto && git add -A && git commit -m "sweep: {tag}" --quiet 2>/dev/null')
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT (>{timeout}s)")
                failed.append(tag + " (timeout)")
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(tag)

elapsed = time.time() - t_start
print(f"\n\n{'='*60}")
print(f"SWEEP COMPLETE in {elapsed:.0f}s")
print(f"  Ran: {done - skipped - len(failed)}, Skipped: {skipped}, Failed: {len(failed)}")
if failed:
    print(f"Failed:")
    for f_item in failed:
        print(f"  - {f_item}")
print(f"Total experiments in log: {len(get_completed_experiments())}")
print(f"{'='*60}")
