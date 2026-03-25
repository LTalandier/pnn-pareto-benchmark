"""
PNN Pareto Benchmark — experiment runner.
THE AGENT EDITS THIS FILE to configure and run experiments.

Usage: python train.py
"""

import torch
from physics import PhotonicNeuralNetwork, TOPOLOGIES
from prepare import prepare_data
from evaluate import (evaluate_model, train_model, log_experiment,
                      get_experiment_count)


# ═══════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION — agent modifies these
# ═══════════════════════════════════════════════

TOPOLOGY = 'clements'       # One of: clements, reck, butterfly, diamond, braid, scf_fractal
MESH_SIZE = 8               # Must be power of 2: 4, 8, 16, 32, 64
DATASET = 'vowel'           # One of: vowel, fmnist, iris
N_PHOTONIC_LAYERS = 1       # Number of stacked photonic layers
LOSS_PER_MZI_DB = 0.2       # Insertion loss per MZI in dB
CLASSIFIER_HIDDEN = (64,)   # Electronic classifier hidden layers
NONLINEARITY = 'photodetect'  # 'photodetect' or 'modReLU'
N_EPOCHS = 100              # Training epochs
LEARNING_RATE = 0.005       # Learning rate
NOISE_AWARE_TRAINING = False  # Enable noise-aware training
TRAIN_NOISE_SIGMA = 0.05   # Noise sigma during training (if noise-aware)

# ═══════════════════════════════════════════════


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {TOPOLOGY} N={MESH_SIZE} on {DATASET}")
    print(f"  Layers={N_PHOTONIC_LAYERS}, Loss={LOSS_PER_MZI_DB}dB, "
          f"Nonlinearity={NONLINEARITY}")
    print(f"  Classifier: {CLASSIFIER_HIDDEN}, Epochs={N_EPOCHS}, "
          f"LR={LEARNING_RATE}")
    if NOISE_AWARE_TRAINING:
        print(f"  Noise-aware training: sigma={TRAIN_NOISE_SIGMA}")
    print(f"{'='*60}\n")

    # Load data
    train_loader, test_loader, data_info = prepare_data(
        DATASET, pca_dim=MESH_SIZE)
    n_classes = data_info['n_classes']

    # Build model
    model = PhotonicNeuralNetwork(
        N=MESH_SIZE,
        topology=TOPOLOGY,
        n_classes=n_classes,
        n_photonic_layers=N_PHOTONIC_LAYERS,
        loss_per_mzi_dB=LOSS_PER_MZI_DB,
        classifier_hidden=CLASSIFIER_HIDDEN,
        nonlinearity=NONLINEARITY,
    ).to(device)

    # Attach metadata for evaluation
    model.N = MESH_SIZE
    model.topology_name = TOPOLOGY

    print(f"Model: {model.get_mzi_count()} MZIs, "
          f"depth {model.get_optical_depth()}")

    # Train
    if NOISE_AWARE_TRAINING:
        from evaluate import train_model_noise_aware
        train_time = train_model_noise_aware(
            model, train_loader, device,
            n_epochs=N_EPOCHS, lr=LEARNING_RATE,
            train_sigma=TRAIN_NOISE_SIGMA)
    else:
        train_time = train_model(
            model, train_loader, device,
            n_epochs=N_EPOCHS, lr=LEARNING_RATE)

    print(f"Training: {train_time:.1f}s")

    # Evaluate
    results = evaluate_model(model, test_loader, device)
    results['training_time_sec'] = train_time

    # Build config dict for logging
    config = {
        'topology': TOPOLOGY,
        'mesh_size': MESH_SIZE,
        'dataset': DATASET,
        'n_photonic_layers': N_PHOTONIC_LAYERS,
        'loss_per_mzi_dB': LOSS_PER_MZI_DB,
        'classifier_hidden': list(CLASSIFIER_HIDDEN),
        'nonlinearity': NONLINEARITY,
        'n_epochs': N_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'noise_aware_training': NOISE_AWARE_TRAINING,
        'train_noise_sigma': TRAIN_NOISE_SIGMA if NOISE_AWARE_TRAINING else 0,
    }

    # Log
    desc = (f"{TOPOLOGY} N={MESH_SIZE} {DATASET} "
            f"layers={N_PHOTONIC_LAYERS} loss={LOSS_PER_MZI_DB}dB")
    if NOISE_AWARE_TRAINING:
        desc += f" noise-aware={TRAIN_NOISE_SIGMA}"
    exp_id = log_experiment(results, description=desc, config=config)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS (experiment {exp_id})")
    print(f"{'='*60}")
    print(f"  Topology:       {TOPOLOGY}")
    print(f"  Mesh size:      {MESH_SIZE}")
    print(f"  MZI count:      {results['n_mzis']}")
    print(f"  Optical depth:  {results['optical_depth']}")
    print(f"  Hardware cost:  {results['hardware_cost']}")
    print(f"  Clean accuracy: {results['clean_accuracy']:.4f}")
    print(f"  Robustness:     {results['robustness']:.4f}")
    print(f"\n  Accuracy by noise level:")
    for sigma, data in sorted(results['accuracies_by_noise'].items(),
                               key=lambda x: float(x[0])):
        print(f"    sigma={sigma}: {data['mean']:.4f} +/- {data.get('std', 0):.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_experiment()
