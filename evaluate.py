"""
Evaluation protocol for PNN Pareto benchmark.
FIXED — the agent must never modify this file.

Records all metrics on separate axes (no composite score).
The Pareto analysis is done post-hoc from the raw data.

Metrics recorded per experiment:
  - topology, mesh_size, dataset
  - n_mzis, optical_depth
  - loss_per_mzi_dB, noise_sigma
  - clean_accuracy (sigma=0)
  - noisy_accuracy_mean, noisy_accuracy_std (at given sigma)
  - robustness (noisy/clean ratio)
  - training_time_sec
"""

import torch
import json
import os
import time
from datetime import datetime, timezone

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "sweep_summary.json")

NOISE_SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
MC_TRIALS = 30  # Monte Carlo trials per noise level


def evaluate_model(model, test_loader, device):
    """
    Full multi-noise evaluation. Returns dict with all metrics.
    """
    model.eval()
    results = {
        'topology': getattr(model, 'topology_name', 'unknown'),
        'mesh_size': model.N if hasattr(model, 'N') else 0,
        'n_mzis': model.get_mzi_count(),
        'optical_depth': model.get_optical_depth(),
        'accuracies_by_noise': {},
    }

    for sigma in NOISE_SIGMAS:
        if sigma == 0.0:
            acc = _compute_accuracy(model, test_loader, device, sigma=0.0)
            results['accuracies_by_noise'][f'{sigma:.3f}'] = {
                'mean': acc, 'std': 0.0, 'trials': 1
            }
            results['clean_accuracy'] = acc
        else:
            trial_accs = _compute_accuracy_mc(
                model, test_loader, device, sigma, MC_TRIALS)
            mean_acc = sum(trial_accs) / len(trial_accs)
            std_acc = torch.tensor(trial_accs).std().item()
            results['accuracies_by_noise'][f'{sigma:.3f}'] = {
                'mean': mean_acc, 'std': std_acc, 'trials': MC_TRIALS
            }

    # Robustness: accuracy retention at worst noise
    worst_sigma = f'{max(NOISE_SIGMAS):.3f}'
    worst_acc = results['accuracies_by_noise'][worst_sigma]['mean']
    clean_acc = results.get('clean_accuracy', 1e-8)
    results['robustness'] = worst_acc / max(clean_acc, 1e-8)

    # Hardware cost (for Pareto analysis)
    results['hardware_cost'] = results['n_mzis'] * results['optical_depth']

    return results


def _compute_accuracy(model, test_loader, device, sigma=0.0):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if sigma > 0:
                preds = model.forward_with_noise(x, sigma)
            else:
                preds = model(x)
            correct += (preds.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def _compute_accuracy_mc(model, test_loader, device, sigma, n_trials):
    """Run n_trials MC noise trials in one vectorized forward pass per batch."""
    trial_correct = torch.zeros(n_trials, device=device)
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size = y.size(0)
            # preds: [n_trials, batch, n_classes]
            preds = model.forward_with_noise(x, sigma, n_mc_trials=n_trials)
            # y: [batch] -> [n_trials, batch]
            y_exp = y.unsqueeze(0).expand(n_trials, -1)
            trial_correct += (preds.argmax(dim=-1) == y_exp).sum(dim=1).float()
            total += batch_size
    return (trial_correct / total).tolist()


def train_model(model, train_loader, device, n_epochs=100, lr=0.005):
    """
    Standard training loop. Returns training time in seconds.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    t_start = time.time()
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # Clamp phases to [0, 2pi]
            with torch.no_grad():
                for mesh in model.meshes:
                    mesh.thetas.data = mesh.thetas.data % (2 * torch.pi)
                    mesh.phis.data = mesh.phis.data % (2 * torch.pi)

    return time.time() - t_start


def train_model_noise_aware(model, train_loader, device,
                            n_epochs=100, lr=0.005, train_sigma=0.05):
    """
    Noise-aware training: inject phase noise during training
    so the network learns robust phase configurations.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    t_start = time.time()
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model.forward(x, noise_sigma=train_sigma)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for mesh in model.meshes:
                    mesh.thetas.data = mesh.thetas.data % (2 * torch.pi)
                    mesh.phis.data = mesh.phis.data % (2 * torch.pi)

    return time.time() - t_start


def log_experiment(results, description="", config=None):
    """Append experiment to JSONL log. Returns experiment ID."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Count existing experiments
    exp_id = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            exp_id = sum(1 for _ in f)

    entry = {
        'id': exp_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': description,
        'config': config or {},
        **{k: v for k, v in results.items() if k != 'accuracies_by_noise'},
        'accuracies_by_noise': results.get('accuracies_by_noise', {}),
    }

    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    print(f"[LOG] Experiment {exp_id}: {description}")
    print(f"  Topology: {results.get('topology', '?')}, "
          f"N={results.get('mesh_size', '?')}, "
          f"MZIs={results.get('n_mzis', '?')}, "
          f"Depth={results.get('optical_depth', '?')}")
    print(f"  Clean acc: {results.get('clean_accuracy', 0):.4f}, "
          f"Robustness: {results.get('robustness', 0):.4f}")

    return exp_id


def get_experiment_count():
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE) as f:
        return sum(1 for _ in f)


def load_all_results():
    """Load all experiment results from log."""
    if not os.path.exists(LOG_FILE):
        return []
    results = []
    with open(LOG_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results
