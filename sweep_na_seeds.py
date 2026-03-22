"""
Noise-aware training seeds: clements + scf_fractal × 3 seeds.
N=16 vowel, CPU.
"""
import subprocess, sys, os, time, textwrap
os.chdir("/root/pnn-pareto")

EXPERIMENTS = [
    ("clements", 1), ("clements", 2), ("clements", 3),
    ("scf_fractal", 1), ("scf_fractal", 2), ("scf_fractal", 3),
]

def write_train_py(topology, seed):
    code = textwrap.dedent(f'''\
        import torch, numpy as np
        from physics import PhotonicNeuralNetwork
        from prepare import prepare_data
        from evaluate import evaluate_model, train_model_noise_aware, log_experiment

        def run_experiment():
            torch.manual_seed({seed})
            np.random.seed({seed})
            device = torch.device("cpu")
            train_loader, test_loader, data_info = prepare_data("vowel", pca_dim=16)
            model = PhotonicNeuralNetwork(N=16, topology="{topology}", n_classes=data_info["n_classes"],
                n_photonic_layers=1, loss_per_mzi_dB=0.2, classifier_hidden=(64,), nonlinearity="photodetect").to(device)
            model.N = 16
            model.topology_name = "{topology}"
            train_time = train_model_noise_aware(model, train_loader, device, n_epochs=100, lr=0.005, train_sigma=0.05)
            results = evaluate_model(model, test_loader, device)
            results["training_time_sec"] = train_time
            config = dict(topology="{topology}", mesh_size=16, dataset="vowel", n_photonic_layers=1,
                loss_per_mzi_dB=0.2, classifier_hidden=[64], nonlinearity="photodetect", n_epochs=100,
                learning_rate=0.005, noise_aware_training=True, train_noise_sigma=0.05, seed={seed})
            desc = "{topology} N=16 vowel noise-aware seed={seed}"
            log_experiment(results, description=desc, config=config)
            print(f"acc={{results['clean_accuracy']:.4f}} rob={{results['robustness']:.4f}}")
        if __name__ == "__main__":
            run_experiment()
    ''')
    with open("train.py", "w") as f:
        f.write(code)

t_start = time.time()
for topo, seed in EXPERIMENTS:
    tag = f"{{topo}} NA seed={{seed}}"
    print(f"\\n{{topo}} noise-aware seed={{seed}}")
    write_train_py(topo, seed)
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=300)
    for line in result.stdout.strip().split("\\n"):
        if "acc=" in line:
            print(f"  {{line.strip()}}")
    if result.returncode != 0:
        print(f"  FAILED")
print(f"\\nDone in {{time.time()-t_start:.0f}}s")
