"""
Crosstalk validation experiment.

Tests whether modeling crossing crosstalk (2x2 coupling) vs pure attenuation
changes butterfly accuracy at N=16. Quick validation for the paper's
Limitation 4 (crossing loss model).

Modifies PhotonicMesh._build_layer_matrix to apply crosstalk coupling
between crossing waveguide pairs instead of scalar attenuation.
"""
import math
import torch
import torch.nn as nn
import numpy as np

from physics import PhotonicMesh, PhotonicNeuralNetwork, get_topology_info
from prepare import prepare_data
from evaluate import evaluate_model, train_model


def get_crossing_pairs(layers_spec, N):
    """
    For each layer, return list of (lo, hi) waveguide pairs that cross.

    When an MZI connects ports (i, j) with |j-i| > 1, the routed signals
    cross each intermediate waveguide. This produces crossings between:
      - port lo and each intermediate port k (lo < k < hi)
      - port hi and each intermediate port k
    But physically, the crossing is between two waveguides passing through
    the same spatial point. For a planar layout, each MZI with span > 1
    creates crossings: the signal from port lo crosses over ports lo+1,
    lo+2, ..., hi-1 sequentially, and signal from port hi crosses the same.

    We model each crossing as a 2x2 coupling between the two waveguides
    that intersect at that point. For MZI (lo, hi), the crossing pairs are:
      (lo, lo+1), (lo, lo+2), ..., (lo, hi-1)  [lo's signal crossing over]
      (hi, hi-1), (hi, hi-2), ..., (hi, lo+1)  [hi's signal crossing over]
      Plus each intermediate pair crossed by both: already counted above.

    Simplification: we model crossings as sequential pairwise swaps.
    For MZI connecting (lo, hi), port lo's signal must cross over
    lo+1, lo+2, ..., hi-1 to reach the MZI. Similarly for hi.
    Each crossing is a 2x2 event between adjacent waveguides during routing.
    So the crossing pairs for MZI (lo, hi) are:
      Before MZI: (lo, lo+1), (lo+1, lo+2), ..., (hi-2, hi-1) for the lo port
                  (hi, hi-1), (hi-1, hi-2), ..., (lo+2, lo+1) for the hi port
    But this double-counts. The simpler model: each intermediate port k
    is crossed once, creating one (k, k+1) or similar crossing event.

    For simplicity and tractability, we use the per-port crossing count
    approach but apply crosstalk as a dense NxN coupling matrix per layer.
    """
    layer_crossing_pairs = []
    for layer in layers_spec:
        pairs = []
        for (pi, pj) in layer:
            lo, hi = min(pi, pj), max(pi, pj)
            if hi - lo > 1:
                # Each intermediate port crosses with its neighbors
                # during the routing of signals to the MZI
                for k in range(lo, hi):
                    pairs.append((k, k + 1))
        # Deduplicate (same pair can be crossed by multiple MZIs)
        pairs = list(set(pairs))
        layer_crossing_pairs.append(pairs)
    return layer_crossing_pairs


def build_crossing_matrix(N, pairs, eps, Lx_dB, device):
    """
    Build NxN crossing coupling matrix for one layer.

    Each crossing pair (i, j) applies:
        [t,  jk] [E_i]
        [jk, t ] [E_j]
    where t = alpha * sqrt(1-eps), k = alpha * sqrt(eps),
    alpha = sqrt(10^(-Lx_dB/10)).

    For ports with multiple crossings, we compose them sequentially.
    """
    alpha = math.sqrt(10 ** (-Lx_dB / 10)) if Lx_dB > 0 else 1.0
    t = alpha * math.sqrt(1 - eps)
    kc = alpha * math.sqrt(eps)

    # Start with identity
    C = torch.eye(N, dtype=torch.complex64, device=device)

    # Apply each crossing as a 2x2 transform
    for (i, j) in pairs:
        C_new = torch.eye(N, dtype=torch.complex64, device=device)
        C_new[i, i] = t
        C_new[j, j] = t
        C_new[i, j] = 1j * kc
        C_new[j, i] = 1j * kc
        C = C @ C_new

    return C


class PhotonicMeshCrosstalk(PhotonicMesh):
    """PhotonicMesh with crosstalk coupling at crossings."""

    def __init__(self, N, topology, loss_per_mzi_dB, crossing_loss_dB, crosstalk_eps):
        # Init parent with crossing_loss_dB=0 (we handle crossings ourselves)
        super().__init__(N, topology, loss_per_mzi_dB, crossing_loss_dB=0.0)
        self.crosstalk_eps = crosstalk_eps
        self._crossing_loss_dB_actual = crossing_loss_dB

        # Compute crossing pairs per layer
        self._crossing_pairs = get_crossing_pairs(self.layers_spec, N)

        # Precompute crossing matrices (they're fixed, not learned)
        self._crossing_matrices = []
        for pairs in self._crossing_pairs:
            if pairs and (crossing_loss_dB > 0 or crosstalk_eps > 0):
                C = build_crossing_matrix(N, pairs, crosstalk_eps,
                                          crossing_loss_dB, torch.device('cpu'))
                self._crossing_matrices.append(C)
            else:
                self._crossing_matrices.append(None)

    def _build_layer_matrix(self, layer_idx, noise_sigma=0.0, n_trials=0):
        """Override: apply crossing as matrix multiply instead of diagonal."""
        # Get the MZI matrix (without crossing loss, since parent has Lx=0)
        M = super()._build_layer_matrix(layer_idx, noise_sigma, n_trials)

        # Apply crossing coupling matrix
        C = self._crossing_matrices[layer_idx]
        if C is not None:
            C = C.to(M.device)
            if M.dim() == 3:  # [n_trials, N, N]
                M = M @ C.unsqueeze(0)
            else:
                M = M @ C
        return M


class PhotonicNeuralNetworkCrosstalk(PhotonicNeuralNetwork):
    """PNN with crosstalk-aware crossing model."""

    def __init__(self, N, topology, n_classes, n_photonic_layers,
                 loss_per_mzi_dB, classifier_hidden, nonlinearity,
                 crossing_loss_dB, crosstalk_eps):
        # Call nn.Module.__init__ directly, skip PhotonicNeuralNetwork.__init__
        nn.Module.__init__(self)
        self.N = N
        self.topology_name = topology
        self.n_photonic_layers = n_photonic_layers
        self.nonlinearity = nonlinearity

        # Photonic layers with crosstalk
        self.meshes = nn.ModuleList([
            PhotonicMeshCrosstalk(N, topology, loss_per_mzi_dB,
                                  crossing_loss_dB, crosstalk_eps)
            for _ in range(n_photonic_layers)
        ])

        # Electronic classifier (copied from parent)
        layers = []
        in_dim = N
        for h in classifier_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.classifier = nn.Sequential(*layers)


def run_experiment(crosstalk_eps, crossing_loss_dB=0.02, seed=None):
    """Run one butterfly N=16 vowel experiment with given crosstalk."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.seed()
        np.random.seed()

    device = torch.device("cpu")
    train_loader, test_loader, data_info = prepare_data("vowel", pca_dim=16)

    model = PhotonicNeuralNetworkCrosstalk(
        N=16, topology="butterfly", n_classes=data_info["n_classes"],
        n_photonic_layers=1, loss_per_mzi_dB=0.2,
        classifier_hidden=(64,), nonlinearity="photodetect",
        crossing_loss_dB=crossing_loss_dB, crosstalk_eps=crosstalk_eps,
    ).to(device)
    model.N = 16
    model.topology_name = "butterfly"

    train_time = train_model(model, train_loader, device, n_epochs=100, lr=0.005)
    results = evaluate_model(model, test_loader, device)
    return results["clean_accuracy"] * 100, results["robustness"]


# ── Main ──────────────────────────────────────────────────────────────

SEEDS = [None, 1, 2, 3]  # canonical seeds matching Tables 3/7/9
CROSSTALK_LEVELS = [
    (0.0,    "0 (atten. only)"),
    (0.0001, "0.0001 (-40 dB)"),
    (0.001,  "0.001 (-30 dB)"),
]

print("Crosstalk validation: butterfly N=16 vowel, Lx=0.02 dB")
print("=" * 65)

all_results = {}
for eps, label in CROSSTALK_LEVELS:
    accs, robs = [], []
    for seed in SEEDS:
        tag = f"seed={seed}" if seed is not None else "seed=default"
        acc, rob = run_experiment(eps, crossing_loss_dB=0.02, seed=seed)
        accs.append(acc)
        robs.append(rob)
        print(f"  eps={label} {tag}: acc={acc:.2f}% rob={rob:.4f}")
    mean_acc = np.mean(accs)
    std_acc = np.std(accs, ddof=0)
    mean_rob = np.mean(robs)
    std_rob = np.std(robs, ddof=0)
    all_results[label] = (mean_acc, std_acc, mean_rob, std_rob)
    print(f"  >> eps={label}: {mean_acc:.1f}±{std_acc:.1f}%, rob={mean_rob:.3f}±{std_rob:.3f}")
    print()

print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Crosstalk (eps)':<25} {'Acc (%)':<15} {'Rob':<15}")
print("-" * 55)
for eps, label in CROSSTALK_LEVELS:
    m_a, s_a, m_r, s_r = all_results[label]
    print(f"{label:<25} {m_a:.1f}±{s_a:.1f}       {m_r:.3f}±{s_r:.3f}")

# Compare with Table 7 baseline
print()
print("Table 7 baseline (Lx=0.02, atten only): 83.6±1.6%, rob=0.379±0.027")
print()

# Decision
baseline = all_results["0 (atten. only)"]
for eps, label in CROSSTALK_LEVELS[1:]:
    shift = abs(all_results[label][0] - baseline[0])
    print(f"Shift for eps={label}: {shift:.1f} pp")
    if shift < 0.5:
        print(f"  -> < 0.5 pp: add sentence to limitations")
    elif shift < 2.0:
        print(f"  -> 0.5-2 pp: add footnote")
    else:
        print(f"  -> > 2 pp: STOP, needs deeper investigation")
