"""
Fmnist N=16 multi-seed: 4 topologies × 3 seeds = 12 runs. CPU. 15 epochs.
"""
import subprocess, sys, os, time, textwrap
os.chdir("/root/pnn-pareto")

TOPOLOGIES = ["clements", "reck", "butterfly", "scf_fractal"]
SEEDS = [1, 2, 3]

def write_train_py(topology, seed):
    code = textwrap.dedent(f'''\
        import torch, numpy as np
        from physics import PhotonicNeuralNetwork
        from prepare import prepare_data
        from evaluate import evaluate_model, train_model, log_experiment

        def run_experiment():
            torch.manual_seed({seed})
            np.random.seed({seed})
            device = torch.device("cpu")
            train_loader, test_loader, data_info = prepare_data("fmnist", pca_dim=16)
            model = PhotonicNeuralNetwork(N=16, topology="{topology}",
                n_classes=data_info["n_classes"], n_photonic_layers=1, loss_per_mzi_dB=0.2,
                classifier_hidden=(64,), nonlinearity="photodetect").to(device)
            model.N = 16
            model.topology_name = "{topology}"
            train_time = train_model(model, train_loader, device, n_epochs=15, lr=0.005)
            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time
            config = dict(topology="{topology}", mesh_size=16, dataset="fmnist",
                n_photonic_layers=1, loss_per_mzi_dB=0.2, classifier_hidden=[64],
                nonlinearity="photodetect", n_epochs=15, learning_rate=0.005,
                noise_aware_training=False, train_noise_sigma=0, seed={seed})
            log_experiment(results, description="{topology} N=16 fmnist seed={seed}", config=config)
            print(f"acc={{results['clean_accuracy']:.4f}} rob={{results['robustness']:.4f}}")
        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)

t_start = time.time()
for topo in TOPOLOGIES:
    for seed in SEEDS:
        print(f"{topo} N=16 fmnist seed={seed}")
        write_train_py(topo, seed)
        r = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=1800)
        for line in r.stdout.strip().split("\n"):
            if "acc=" in line: print(f"  {line.strip()}")
        if r.returncode != 0: print(f"  FAILED")
print(f"\nDone in {time.time()-t_start:.0f}s")
