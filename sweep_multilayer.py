"""
Multi-layer test: 2-layer butterfly and Clements at N=16, vowel, modReLU, 4 seeds each.
"""
import subprocess, sys, os, time, textwrap
os.chdir("/root/pnn-pareto")

EXPERIMENTS = []
for topo in ["butterfly", "clements"]:
    for seed in [0, 1, 2, 3]:
        EXPERIMENTS.append((topo, seed))

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
            train_loader, test_loader, data_info = prepare_data("vowel", pca_dim=16)
            model = PhotonicNeuralNetwork(N=16, topology="{topology}",
                n_classes=data_info["n_classes"], n_photonic_layers=2, loss_per_mzi_dB=0.2,
                classifier_hidden=(64,), nonlinearity="modReLU").to(device)
            model.N = 16
            model.topology_name = "{topology}"
            train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time
            config = dict(topology="{topology}", mesh_size=16, dataset="vowel",
                n_photonic_layers=2, loss_per_mzi_dB=0.2, classifier_hidden=[64],
                nonlinearity="modReLU", n_epochs=100, learning_rate=0.005,
                noise_aware_training=False, train_noise_sigma=0, seed={seed})
            log_experiment(results, description="{topology} N=16 vowel 2-layer modReLU seed={seed}", config=config)
            print(f"acc={{results['clean_accuracy']:.4f}} rob={{results['robustness']:.4f}} mzis={{model.get_mzi_count()}} depth={{model.get_optical_depth()}}")
        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)

t_start = time.time()
for topo, seed in EXPERIMENTS:
    print(f"{topo} 2-layer modReLU N=16 vowel seed={seed}")
    write_train_py(topo, seed)
    r = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=600)
    for line in r.stdout.strip().split("\n"):
        if "acc=" in line: print(f"  {line.strip()}")
    if r.returncode != 0: print(f"  FAILED: {r.stderr[-200:]}")
print(f"\nDone in {time.time()-t_start:.0f}s")
