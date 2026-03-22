"""
Statistical validity sweep: 4 topologies × 3 loss levels × 3 seeds = 36 experiments.
All at N=16, vowel, CPU.
Seeds 1, 2, 3 (we already have seed=default for 0.2 dB).
"""
import subprocess
import sys
import os
import time
import textwrap
import json

os.chdir("/root/pnn-pareto")

TOPOLOGIES = ["clements", "reck", "butterfly", "scf_fractal"]
LOSS_LEVELS = [0.2, 0.3, 0.5]
SEEDS = [1, 2, 3]

def write_train_py(topology, loss_per_mzi_dB, seed):
    code = textwrap.dedent(f'''\
        import torch
        import numpy as np
        from physics import PhotonicNeuralNetwork, TOPOLOGIES
        from prepare import prepare_data
        from evaluate import (evaluate_model, train_model, log_experiment,
                              get_experiment_count)

        TOPOLOGY = "{topology}"
        MESH_SIZE = 16
        DATASET = "vowel"
        N_PHOTONIC_LAYERS = 1
        LOSS_PER_MZI_DB = {loss_per_mzi_dB}
        CLASSIFIER_HIDDEN = (64,)
        NONLINEARITY = "photodetect"
        N_EPOCHS = 100
        LEARNING_RATE = 0.005
        SEED = {seed}

        def run_experiment():
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            device = torch.device("cpu")
            print(f"EXPERIMENT: {{TOPOLOGY}} N=16 vowel loss={{LOSS_PER_MZI_DB}}dB seed={{SEED}} [{{device}}]")

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
                seed=SEED,
            )

            desc = f"{{TOPOLOGY}} N=16 vowel loss={{LOSS_PER_MZI_DB}}dB seed={{SEED}}"
            exp_id = log_experiment(results, description=desc, config=config)
            print(f"RESULT: clean_acc={{results['clean_accuracy']:.4f}} robustness={{results['robustness']:.4f}}")

        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)


experiments = []
for loss in LOSS_LEVELS:
    for topo in TOPOLOGIES:
        for seed in SEEDS:
            experiments.append(dict(
                topology=topo, loss=loss, seed=seed,
                tag=f"{topo} N=16 vowel loss={loss}dB seed={seed}"
            ))

total = len(experiments)
failed = []
print(f"=== SEED SWEEP: {total} experiments ===\n")
t_start = time.time()

for i, exp in enumerate(experiments):
    tag = exp["tag"]
    print(f"\n[{i+1}/{total}] {tag}")
    print("-" * 50)

    write_train_py(exp["topology"], exp["loss"], exp["seed"])

    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True, timeout=900)
        print(result.stdout[-200:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-200:]}")
            failed.append(tag)
        else:
            os.system(f'cd /root/pnn-pareto && git add -A && git commit -m "seed: {tag}" --quiet 2>/dev/null')
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        failed.append(tag + " (timeout)")

elapsed = time.time() - t_start
print(f"\n\n{'='*60}")
print(f"SEED SWEEP COMPLETE in {elapsed:.0f}s")
print(f"  Ran: {total - len(failed)}, Failed: {len(failed)}")
if failed:
    print("Failed:")
    for f_item in failed:
        print(f"  - {f_item}")
print(f"{'='*60}")

# Now compute summary statistics
print(f"\n\n{'='*60}")
print("SUMMARY: mean +/- std across seeds")
print(f"{'='*60}\n")

# Load all results
results = []
with open("results/experiment_log.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

# Filter to N=16 vowel seed experiments + original runs
import numpy as np_stats

for loss in LOSS_LEVELS:
    print(f"\n--- Loss = {loss} dB ---")
    print(f"{'Topology':12s}  {'Acc (mean +/- std)':20s}  {'Rob (mean +/- std)':20s}  {'Acc@0.1 (mean+/-std)':20s}")
    for topo in TOPOLOGIES:
        # Get all matching experiments
        matching = [r for r in results
                    if r.get("config", {}).get("topology") == topo
                    and r.get("config", {}).get("mesh_size") == 16
                    and r.get("config", {}).get("dataset") == "vowel"
                    and abs(r.get("config", {}).get("loss_per_mzi_dB", 0.2) - loss) < 0.01
                    and not r.get("config", {}).get("noise_aware_training", False)]

        if not matching:
            print(f"{topo:12s}  NO DATA")
            continue

        accs = [r["clean_accuracy"] for r in matching]
        robs = [r["robustness"] for r in matching]
        acc01s = [r["accuracies_by_noise"]["0.100"]["mean"] for r in matching]

        acc_mean = np_stats.mean(accs)
        acc_std = np_stats.std(accs)
        rob_mean = np_stats.mean(robs)
        rob_std = np_stats.std(robs)
        a01_mean = np_stats.mean(acc01s)
        a01_std = np_stats.std(acc01s)

        print(f"{topo:12s}  {acc_mean:.4f} +/- {acc_std:.4f}   {rob_mean:.4f} +/- {rob_std:.4f}   {a01_mean:.4f} +/- {a01_std:.4f}   (n={len(matching)})")

print(f"\n{'='*60}")
print("KEY CHECKS:")
print("1. Reck robustness anomaly: does Reck (depth 29) still beat Clements (depth 16)?")
reck_02 = [r["robustness"] for r in results
           if r.get("config",{}).get("topology") == "reck"
           and r.get("config",{}).get("mesh_size") == 16
           and abs(r.get("config",{}).get("loss_per_mzi_dB",0.2) - 0.2) < 0.01
           and not r.get("config",{}).get("noise_aware_training", False)]
clem_02 = [r["robustness"] for r in results
           if r.get("config",{}).get("topology") == "clements"
           and r.get("config",{}).get("mesh_size") == 16
           and abs(r.get("config",{}).get("loss_per_mzi_dB",0.2) - 0.2) < 0.01
           and not r.get("config",{}).get("noise_aware_training", False)]
print(f"  Reck robustness:     {np_stats.mean(reck_02):.4f} +/- {np_stats.std(reck_02):.4f}")
print(f"  Clements robustness: {np_stats.mean(clem_02):.4f} +/- {np_stats.std(clem_02):.4f}")
if np_stats.mean(reck_02) > np_stats.mean(clem_02) + np_stats.std(clem_02):
    print("  -> Anomaly HOLDS (Reck mean > Clements mean + 1 std)")
else:
    print("  -> Anomaly may be within variance")

print("\n2. Table III non-monotonicity check:")
for topo in TOPOLOGIES:
    means = {}
    for loss in LOSS_LEVELS:
        matching = [r["clean_accuracy"] for r in results
                    if r.get("config",{}).get("topology") == topo
                    and r.get("config",{}).get("mesh_size") == 16
                    and abs(r.get("config",{}).get("loss_per_mzi_dB",0.2) - loss) < 0.01
                    and not r.get("config",{}).get("noise_aware_training", False)]
        means[loss] = np_stats.mean(matching) if matching else 0
    monotonic = means[0.2] >= means[0.3] >= means[0.5]
    status = "OK (monotonic)" if monotonic else "NON-MONOTONIC"
    print(f"  {topo:12s}: 0.2dB={means[0.2]:.4f}  0.3dB={means[0.3]:.4f}  0.5dB={means[0.5]:.4f}  {status}")

print(f"\n{'='*60}")
