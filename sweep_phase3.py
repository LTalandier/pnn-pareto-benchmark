"""
Phase 3: Targeted follow-up experiments (12 total).
All on CPU.
"""
import subprocess
import sys
import os
import time
import textwrap

os.chdir("/root/pnn-pareto")

def write_train_py(topology, mesh_size, dataset, n_epochs=100,
                   loss_per_mzi_dB=0.2, noise_aware=False, train_sigma=0.05):
    code = textwrap.dedent(f'''\
        import torch
        from physics import PhotonicNeuralNetwork, TOPOLOGIES
        from prepare import prepare_data
        from evaluate import (evaluate_model, train_model, train_model_noise_aware,
                              log_experiment, get_experiment_count)

        TOPOLOGY = "{topology}"
        MESH_SIZE = {mesh_size}
        DATASET = "{dataset}"
        N_PHOTONIC_LAYERS = 1
        LOSS_PER_MZI_DB = {loss_per_mzi_dB}
        CLASSIFIER_HIDDEN = (64,)
        NONLINEARITY = "photodetect"
        N_EPOCHS = {n_epochs}
        LEARNING_RATE = 0.005
        NOISE_AWARE_TRAINING = {noise_aware}
        TRAIN_NOISE_SIGMA = {train_sigma}

        def run_experiment():
            device = torch.device("cpu")
            print(f"EXPERIMENT: {{TOPOLOGY}} N={{MESH_SIZE}} on {{DATASET}} [{{device}}]")
            print(f"  loss={{LOSS_PER_MZI_DB}}dB noise_aware={{NOISE_AWARE_TRAINING}}")

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

            if NOISE_AWARE_TRAINING:
                train_time = train_model_noise_aware(
                    model, train_loader, device,
                    n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                    train_sigma=TRAIN_NOISE_SIGMA)
            else:
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
                noise_aware_training=NOISE_AWARE_TRAINING,
                train_noise_sigma=TRAIN_NOISE_SIGMA if NOISE_AWARE_TRAINING else 0,
            )

            desc = f"{{TOPOLOGY}} N={{MESH_SIZE}} {{DATASET}} loss={{LOSS_PER_MZI_DB}}dB"
            if NOISE_AWARE_TRAINING:
                desc += f" noise-aware={{TRAIN_NOISE_SIGMA}}"
            exp_id = log_experiment(results, description=desc, config=config)
            print(f"RESULT: clean_acc={{results['clean_accuracy']:.4f}} robustness={{results['robustness']:.4f}}")

        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)


# Define all 12 experiments
experiments = []

# 1. Insertion loss sweep: 4 topologies × 2 loss values at N=16 vowel
for topo in ["clements", "reck", "butterfly", "scf_fractal"]:
    for loss in [0.3, 0.5]:
        experiments.append(dict(
            topology=topo, mesh_size=16, dataset="vowel",
            n_epochs=100, loss_per_mzi_dB=loss,
            noise_aware=False, train_sigma=0.05,
            tag=f"{topo} N=16 vowel loss={loss}dB"
        ))

# 2. Noise-aware training: clements and scf_fractal at N=16 vowel
for topo in ["clements", "scf_fractal"]:
    experiments.append(dict(
        topology=topo, mesh_size=16, dataset="vowel",
        n_epochs=100, loss_per_mzi_dB=0.2,
        noise_aware=True, train_sigma=0.05,
        tag=f"{topo} N=16 vowel noise-aware"
    ))

# 3. N=32 vowel: clements and butterfly
for topo in ["clements", "butterfly"]:
    experiments.append(dict(
        topology=topo, mesh_size=32, dataset="vowel",
        n_epochs=100, loss_per_mzi_dB=0.2,
        noise_aware=False, train_sigma=0.05,
        tag=f"{topo} N=32 vowel"
    ))

total = len(experiments)
failed = []
print(f"=== PHASE 3: {total} experiments ===\n")
t_start = time.time()

for i, exp in enumerate(experiments):
    tag = exp.pop("tag")
    print(f"\n[{i+1}/{total}] {tag}")
    print("-" * 50)

    write_train_py(**exp)

    timeout = 1800 if exp["mesh_size"] >= 32 else 900
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True, timeout=timeout)
        print(result.stdout[-300:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-300:]}")
            failed.append(tag)
        else:
            os.system(f'cd /root/pnn-pareto && git add -A && git commit -m "phase3: {tag}" --quiet 2>/dev/null')
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>{timeout}s)")
        failed.append(tag + " (timeout)")
    except Exception as e:
        print(f"  ERROR: {e}")
        failed.append(tag)

elapsed = time.time() - t_start
print(f"\n\n{'='*60}")
print(f"PHASE 3 COMPLETE in {elapsed:.0f}s")
print(f"  Ran: {total - len(failed)}, Failed: {len(failed)}")
if failed:
    print("Failed:")
    for f_item in failed:
        print(f"  - {f_item}")
print(f"{'='*60}")
