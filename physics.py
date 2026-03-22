"""
MZI physics primitives and mesh topology generators.
FIXED — the agent must never modify this file.

Topologies implemented:
  1. Clements (rectangular) — universal, N(N-1)/2 MZIs, depth N
  2. Reck (triangular) — universal, N(N-1)/2 MZIs, depth 2N-3
  3. Butterfly/FFT — compact, N/2*log2(N) MZIs, depth log2(N)
  4. Diamond — balanced paths, ~N(N-1)/2 MZIs, depth ~N
  5. Braid — perfectly balanced, ~N(N-1)/2 MZIs
  6. SCF Fractal — recursive CS decomposition, N(N-1)/2 MZIs

References:
  Clements et al., Optica 3, 1460 (2016)
  Reck et al., PRL 73, 58 (1994)
  Fldzhyan et al., Optics Express (2024); Tian et al., Nanophotonics (2022)
  Shokraneh et al., Optics Express 28, 23495 (2020)
  Marchesin et al., Optics Express 33 (2025)
  Basani et al., Nanophotonics 12, 5 (2023)
"""

import torch
import torch.nn as nn
import math
import numpy as np


# ─────────────────────────────────────────────
#  MZI primitives
# ─────────────────────────────────────────────

def mzi_transfer_matrix(theta, phi):
    """
    Single MZI 2x2 unitary.
    U(θ,φ) = [[e^{iφ} cos θ, -sin θ],
               [e^{iφ} sin θ,  cos θ]]
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    exp_phi = torch.exp(1j * phi)
    U = torch.stack([
        torch.stack([exp_phi * cos_t, -sin_t], dim=-1),
        torch.stack([exp_phi * sin_t, cos_t], dim=-1)
    ], dim=-2)
    return U


def apply_mzi(state, U, port_i, port_j):
    """Apply 2x2 MZI to ports i,j of state vector [batch, N]."""
    new_state = state.clone()
    pair = torch.stack([state[..., port_i], state[..., port_j]], dim=-1)
    result = torch.einsum('...ij,...j->...i', U, pair)
    new_state[..., port_i] = result[..., 0]
    new_state[..., port_j] = result[..., 1]
    return new_state


def photodetect(amplitudes):
    """Square-law detection: I = |E|^2"""
    return torch.abs(amplitudes) ** 2


def modReLU(z, bias):
    """modReLU(z) = (|z| + b) * z/|z| if |z| + b > 0, else 0"""
    mag = torch.abs(z)
    phase = z / (mag + 1e-8)
    return torch.relu(mag + bias) * phase


# ─────────────────────────────────────────────
#  Topology generators
#  Each returns: list of layers, where each layer
#  is a list of (port_i, port_j) tuples.
#  MZIs in the same layer execute in parallel.
# ─────────────────────────────────────────────

def clements_topology(N):
    """
    Clements rectangular mesh.
    N(N-1)/2 MZIs, depth N. Universal.
    Alternating even/odd columns of adjacent-pair MZIs.
    """
    layers = []
    for col in range(N):
        layer = []
        if col % 2 == 0:
            for i in range(0, N - 1, 2):
                layer.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                layer.append((i, i + 1))
        if layer:
            layers.append(layer)
    return layers


def reck_topology(N):
    """
    Reck triangular mesh.
    N(N-1)/2 MZIs, depth 2N-3. Universal.
    Triangular elimination: each column eliminates one matrix element.
    Column k applies one MZI; columns targeting non-overlapping ports
    can be merged into the same layer.
    """
    # Generate the sequence of (port_i, port_j) for each column
    # Reck eliminates column-by-column of the unitary matrix.
    # For an NxN unitary, we eliminate elements in order:
    #   row N-1 cols 0..N-2, then row N-2 cols 0..N-3, etc.
    # Each elimination uses one MZI on adjacent ports.
    mzi_sequence = []
    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            mzi_sequence.append((row - 1, row))

    # Pack into layers: MZIs on non-overlapping ports go in same layer
    layers = []
    for pair in mzi_sequence:
        placed = False
        for layer in layers:
            # Check if this pair conflicts with any existing pair in layer
            conflict = False
            for existing in layer:
                if pair[0] in existing or pair[1] in existing:
                    conflict = True
                    break
            if not conflict:
                layer.append(pair)
                placed = True
                break
        if not placed:
            layers.append([pair])
    return layers


def butterfly_topology(N):
    """
    Butterfly/FFT mesh.
    N/2 * log2(N) MZIs, depth log2(N). NOT universal.
    Stride-based connections following FFT butterfly pattern.
    Requires N to be a power of 2.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"N={N} must be power of 2"
    n_stages = int(math.log2(N))
    layers = []
    for stage in range(n_stages):
        layer = []
        stride = 1 << stage
        for block_start in range(0, N, 2 * stride):
            for k in range(stride):
                i = block_start + k
                j = block_start + k + stride
                if j < N:
                    layer.append((i, j))
        if layer:
            layers.append(layer)
    return layers



# Diamond and Braid topologies removed — correct implementations
# require non-trivial wiring patterns from the original papers
# (Shokraneh et al. 2020, Marchesin et al. 2025).
# To be added after paper review.


def scf_fractal_topology(N):
    """
    Self-similar Cosine-Sine Fractal (Basani et al., Nanophotonics, 2023).
    Recursive CS decomposition. N(N-1)/2 MZIs.
    Error scales as O(N * sigma^4) vs O(N^2 * sigma^2) for Clements.

    Implementation: recursively decompose N-port unitary into
    two N/2-port unitaries connected by a mixing layer of N/2 MZIs.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"N={N} must be power of 2"

    def _scf_recursive(ports):
        """Generate layers for a subset of ports."""
        n = len(ports)
        if n <= 1:
            return []
        if n == 2:
            return [[(ports[0], ports[1])]]

        half = n // 2
        top_ports = ports[:half]
        bot_ports = ports[half:]

        # Left sub-block
        left_layers = _scf_recursive(top_ports)
        # Right sub-block
        right_layers = _scf_recursive(bot_ports)

        # Mixing layer: connect top[i] with bot[i]
        mixing = [(top_ports[i], bot_ports[i]) for i in range(half)]

        # Combine: left, then mixing, then right
        # Pad shorter sub-block with empty layers
        max_sub = max(len(left_layers), len(right_layers))
        combined = []

        # Left and right sub-blocks can run in parallel (different ports)
        for i in range(max_sub):
            layer = []
            if i < len(left_layers):
                layer.extend(left_layers[i])
            if i < len(right_layers):
                layer.extend(right_layers[i])
            combined.append(layer)

        # Mixing layer
        combined.append(mixing)

        # Second pass of sub-blocks (for full CS decomposition)
        left_layers2 = _scf_recursive(top_ports)
        right_layers2 = _scf_recursive(bot_ports)
        max_sub2 = max(len(left_layers2), len(right_layers2))
        for i in range(max_sub2):
            layer = []
            if i < len(left_layers2):
                layer.extend(left_layers2[i])
            if i < len(right_layers2):
                layer.extend(right_layers2[i])
            combined.append(layer)

        return combined

    ports = list(range(N))
    return _scf_recursive(ports)


# Registry of all topologies
TOPOLOGIES = {
    'clements': clements_topology,
    'reck': reck_topology,
    'butterfly': butterfly_topology,
    'scf_fractal': scf_fractal_topology,
}


def get_topology_info(name, N):
    """Return (layers, n_mzis, optical_depth) for a topology."""
    fn = TOPOLOGIES[name]
    layers = fn(N)
    n_mzis = sum(len(layer) for layer in layers)
    optical_depth = len(layers)
    return layers, n_mzis, optical_depth


# ─────────────────────────────────────────────
#  PhotonicMesh module — the core differentiable layer
# ─────────────────────────────────────────────

class PhotonicMesh(nn.Module):
    """
    Differentiable MZI mesh with insertion loss.

    Forward pass:
      1. Encode real input as complex amplitudes
      2. Apply MZI layers sequentially
      3. Apply insertion loss after each MZI
      4. Photodetect: |output|^2

    Args:
        N: mesh size (number of ports/waveguides)
        topology: topology name string (key in TOPOLOGIES)
        loss_per_mzi_dB: insertion loss per MZI in dB (typical: 0.2-0.5)
    """
    def __init__(self, N, topology='clements', loss_per_mzi_dB=0.2):
        super().__init__()
        self.N = N
        self.topology_name = topology
        self.loss_per_mzi_dB = loss_per_mzi_dB

        self.layers_spec, self.n_mzis, self.optical_depth = \
            get_topology_info(topology, N)

        # Learnable phases: one (theta, phi) pair per MZI
        self.thetas = nn.Parameter(torch.rand(self.n_mzis) * 2 * math.pi)
        self.phis = nn.Parameter(torch.rand(self.n_mzis) * 2 * math.pi)

        # Insertion loss attenuation (amplitude, not power)
        self._atten = math.sqrt(10 ** (-loss_per_mzi_dB / 10))

    def forward(self, x, noise_sigma=0.0):
        """
        Args:
            x: real input [batch, N]
            noise_sigma: phase noise std in radians (0 = no noise)
        Returns:
            intensities: [batch, N] after photodetection
        """
        state = x.to(torch.complex64)
        mzi_idx = 0

        for layer in self.layers_spec:
            for (pi, pj) in layer:
                theta = self.thetas[mzi_idx]
                phi = self.phis[mzi_idx]

                if noise_sigma > 0:
                    theta = theta + noise_sigma * torch.randn_like(theta)
                    phi = phi + noise_sigma * torch.randn_like(phi)

                U = mzi_transfer_matrix(theta, phi)
                state = apply_mzi(state, U, pi, pj)

                # Insertion loss on the two ports that traversed this MZI
                state = state.clone()
                state[..., pi] = state[..., pi] * self._atten
                state[..., pj] = state[..., pj] * self._atten

                mzi_idx += 1

        return photodetect(state)

    def get_mzi_count(self):
        return self.n_mzis

    def get_optical_depth(self):
        return self.optical_depth

    def get_phase_params(self):
        return self.thetas, self.phis


class PhotonicNeuralNetwork(nn.Module):
    """
    Complete PNN: photonic mesh(es) + electronic classifier.

    Architecture:
      input -> PhotonicMesh -> [nonlinearity -> PhotonicMesh ->]* -> classifier

    Args:
        N: mesh size
        topology: topology name
        n_classes: number of output classes
        n_photonic_layers: number of stacked photonic layers
        loss_per_mzi_dB: insertion loss per MZI
        classifier_hidden: hidden layer sizes for electronic classifier
        nonlinearity: 'photodetect' (default), 'modReLU', or 'abs_sq'
    """
    def __init__(self, N, topology='clements', n_classes=10,
                 n_photonic_layers=1, loss_per_mzi_dB=0.2,
                 classifier_hidden=(64,), nonlinearity='photodetect'):
        super().__init__()
        self.N = N
        self.topology_name = topology
        self.n_photonic_layers = n_photonic_layers
        self.nonlinearity = nonlinearity

        # Photonic layers
        self.meshes = nn.ModuleList([
            PhotonicMesh(N, topology, loss_per_mzi_dB)
            for _ in range(n_photonic_layers)
        ])

        # modReLU biases (if used)
        if nonlinearity == 'modReLU':
            self.modrelu_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(N) - 0.5)
                for _ in range(max(n_photonic_layers - 1, 0))
            ])

        # Electronic classifier
        layers = []
        in_dim = N
        for h in classifier_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x, noise_sigma=0.0):
        """Forward pass through photonic + electronic layers."""
        state = x

        for i, mesh in enumerate(self.meshes):
            if self.nonlinearity == 'modReLU' and i > 0:
                # Re-encode as complex for next photonic layer
                state_complex = state.to(torch.complex64)
                state_complex = modReLU(state_complex, self.modrelu_biases[i-1])
                state = mesh(state_complex.real, noise_sigma)
            else:
                state = mesh(state, noise_sigma)

        return self.classifier(state)

    def forward_with_noise(self, x, sigma):
        return self.forward(x, noise_sigma=sigma)

    def get_mzi_count(self):
        return sum(m.get_mzi_count() for m in self.meshes)

    def get_optical_depth(self):
        return sum(m.get_optical_depth() for m in self.meshes)

    def get_phase_params(self):
        all_thetas = torch.cat([m.thetas for m in self.meshes])
        all_phis = torch.cat([m.phis for m in self.meshes])
        return all_thetas, all_phis
