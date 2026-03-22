"""
Fmnist multi-seed: 4 topologies × N=4,8 × 3 seeds = 24 runs. CPU.
"""
import subprocess, sys, os, time, textwrap
os.chdir("/root/pnn-pareto")

TOPOLOGIES = ["clements", "reck", "butterfly", "scf_fractal"]
MESH_SIZES = [4, 8]
SEEDS = [1, 2, 3]

def write_train_py(topology, mesh_size, seed):
    n_epochs = 30
    code = textwrap.dedent(f'''\
        import torch, numpy as np
        from physics import PhotonicNeuralNetwork
        from prepare import prepare_data
        from evaluate import evaluate_model, train_model, log_experiment

        def run_experiment():
            torch.manual_seed({seed})
            np.random.seed({seed})
            device = torch.device("cpu")
            train_loader, test_loader, data_info = prepare_data("fmnist", pca_dim={mesh_size})
            model = PhotonicNeuralNetwork(N={mesh_size}, topology="{topology}",
                n_classes=data_info["n_classes"], n_photonic_layers=1, loss_per_mzi_dB=0.2,
                classifier_hidden=(64,), nonlinearity="photodetect").to(device)
            model.N = {mesh_size}
            model.topology_name = "{topology}"
            train_time = train_model(model, train_loader, device, n_epochs={n_epochs}, lr=0.005)
            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time
            config = dict(topology="{topology}", mesh_size={mesh_size}, dataset="fmnist",
                n_photonic_layers=1, loss_per_mzi_dB=0.2, classifier_hidden=[64],
                nonlinearity="photodetect", n_epochs={n_epochs}, learning_rate=0.005,
                noise_aware_training=False, train_noise_sigma=0, seed={seed})
            log_experiment(results, description="{topology} N={mesh_size} fmnist seed={seed}", config=config)
            print(f"acc={{results['clean_accuracy']:.4f}} rob={{results['robustness']:.4f}}")
        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)

t_start = time.time()
for ms in MESH_SIZES:
    for topo in TOPOLOGIES:
        for seed in SEEDS:
            print(f"{topo} N={ms} fmnist seed={seed}")
            write_train_py(topo, ms, seed)
            r = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=600)
            for line in r.stdout.strip().split("\n"):
                if "acc=" in line: print(f"  {line.strip()}")
            if r.returncode != 0: print(f"  FAILED")
print(f"\nDone in {time.time()-t_start:.0f}s")
