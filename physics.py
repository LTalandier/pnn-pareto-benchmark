"""
MZI physics primitives and mesh topology generators.

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
    def __init__(self, N, topology='clements', loss_per_mzi_dB=0.2,
                 crossing_loss_dB=0.0):
        super().__init__()
        self.N = N
        self.topology_name = topology
        self.loss_per_mzi_dB = loss_per_mzi_dB
        self.crossing_loss_dB = crossing_loss_dB

        self.layers_spec, self.n_mzis, self.optical_depth = \
            get_topology_info(topology, N)

        # Learnable phases: one (theta, phi) pair per MZI
        self.thetas = nn.Parameter(torch.rand(self.n_mzis) * 2 * math.pi)
        self.phis = nn.Parameter(torch.rand(self.n_mzis) * 2 * math.pi)

        # Insertion loss attenuation (amplitude, not power)
        self._atten = math.sqrt(10 ** (-loss_per_mzi_dB / 10))

        # Precompute per-layer metadata for batched forward pass
        self._layer_mzi_ranges = []  # (start, end) index into thetas/phis
        self._layer_port_i = []      # LongTensor of port_i per layer
        self._layer_port_j = []      # LongTensor of port_j per layer
        # Bitmask of ports NOT touched by MZIs in each layer (for identity diagonal)
        self._layer_passthrough = []  # LongTensor of passthrough port indices

        # Precompute per-layer crossing attenuation [N] per layer.
        # A crossing occurs when an MZI connects (i,j) with |i-j|>1;
        # every port k strictly between i and j is crossed by that MZI.
        self._layer_crossing_atten = []  # list of Tensor[N] amplitude scale
        self._total_crossings = 0  # total across all layers (for reporting)

        mzi_idx = 0
        for layer in self.layers_spec:
            n_in_layer = len(layer)
            start = mzi_idx
            end = mzi_idx + n_in_layer
            self._layer_mzi_ranges.append((start, end))

            pi_list = [p[0] for p in layer]
            pj_list = [p[1] for p in layer]
            self._layer_port_i.append(torch.tensor(pi_list, dtype=torch.long))
            self._layer_port_j.append(torch.tensor(pj_list, dtype=torch.long))

            touched = set(pi_list + pj_list)
            passthrough = [p for p in range(N) if p not in touched]
            self._layer_passthrough.append(torch.tensor(passthrough, dtype=torch.long))

            # Count crossings per port in this layer.
            # For MZI connecting (i, j) with span = |j-i|:
            #   - Ports i and j each cross (span-1) intermediate waveguides
            #     to reach the MZI
            #   - Each intermediate port k (i < k < j) is crossed over once
            crossing_count = [0] * N
            for (pi, pj) in layer:
                lo, hi = min(pi, pj), max(pi, pj)
                span = hi - lo
                if span > 1:
                    # MZI ports: routed signals cross intermediate waveguides
                    crossing_count[lo] += span - 1
                    crossing_count[hi] += span - 1
                    # Intermediate ports: crossed over by the routed signals
                    for k in range(lo + 1, hi):
                        crossing_count[k] += 1
            self._total_crossings += sum(crossing_count)

            # Convert to amplitude attenuation vector
            if crossing_loss_dB > 0.0:
                cross_atten_per = math.sqrt(10 ** (-crossing_loss_dB / 10))
                atten_vec = torch.tensor(
                    [cross_atten_per ** c for c in crossing_count],
                    dtype=torch.float32)
            else:
                atten_vec = torch.ones(N, dtype=torch.float32)
            self._layer_crossing_atten.append(atten_vec)

            mzi_idx += n_in_layer

        self._has_crossing_loss = crossing_loss_dB > 0.0
        self._cached_matrix = None
        self._n_layers = len(self.layers_spec)

    def _build_layer_matrix(self, layer_idx, noise_sigma=0.0, n_trials=0):
        """
        Build NxN transfer matrix for one layer of parallel MZIs.
        Includes insertion loss and crossing loss in the matrix elements.

        When n_trials > 0 and noise_sigma > 0, returns [n_trials, N, N].
        Otherwise returns [N, N].
        """
        start, end = self._layer_mzi_ranges[layer_idx]
        thetas = self.thetas[start:end]
        phis = self.phis[start:end]
        port_i = self._layer_port_i[layer_idx]
        port_j = self._layer_port_j[layer_idx]
        passthrough = self._layer_passthrough[layer_idx]
        dev = thetas.device

        if n_trials > 0 and noise_sigma > 0:
            n_mzis = end - start
            # [n_trials, n_mzis] independent noise
            thetas = thetas.unsqueeze(0) + noise_sigma * torch.randn(
                n_trials, n_mzis, device=dev)
            phis = phis.unsqueeze(0) + noise_sigma * torch.randn(
                n_trials, n_mzis, device=dev)

            cos_t = torch.cos(thetas)
            sin_t = torch.sin(thetas)
            exp_phi = torch.exp(1j * phis)
            atten = self._atten

            # Build [n_trials, N, N] matrix
            M = torch.zeros(n_trials, self.N, self.N,
                            dtype=torch.complex64, device=dev)
            # Passthrough ports get identity
            if len(passthrough) > 0:
                M[:, passthrough, passthrough] = 1.0
            # MZI blocks (with attenuation folded in)
            # state @ M convention: M[j,k] maps input j to output k
            # new[port_i] = U00 * old[port_i] + U01 * old[port_j]
            # -> M[port_i, port_i] = U00, M[port_j, port_i] = U01
            M[:, port_i, port_i] = (exp_phi * cos_t * atten).to(torch.complex64)
            M[:, port_j, port_i] = (-sin_t * atten).to(torch.complex64)
            M[:, port_i, port_j] = (exp_phi * sin_t * atten).to(torch.complex64)
            M[:, port_j, port_j] = (cos_t * atten).to(torch.complex64)

            # Apply crossing loss: scale each output port (column) by its
            # crossing attenuation.  M = M @ diag(cross_atten)
            if self._has_crossing_loss:
                ca = self._layer_crossing_atten[layer_idx].to(dev)
                M = M * ca.to(torch.complex64).unsqueeze(0).unsqueeze(0)
            return M

        # Single matrix path
        if noise_sigma > 0:
            thetas = thetas + noise_sigma * torch.randn_like(thetas)
            phis = phis + noise_sigma * torch.randn_like(phis)

        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)
        exp_phi = torch.exp(1j * phis)
        atten = self._atten

        M = torch.zeros(self.N, self.N, dtype=torch.complex64, device=dev)
        if len(passthrough) > 0:
            M[passthrough, passthrough] = 1.0
        M[port_i, port_i] = (exp_phi * cos_t * atten).to(torch.complex64)
        M[port_j, port_i] = (-sin_t * atten).to(torch.complex64)
        M[port_i, port_j] = (exp_phi * sin_t * atten).to(torch.complex64)
        M[port_j, port_j] = (cos_t * atten).to(torch.complex64)

        # Apply crossing loss: scale each output port (column)
        if self._has_crossing_loss:
            ca = self._layer_crossing_atten[layer_idx].to(dev)
            M = M * ca.to(torch.complex64).unsqueeze(0)
        return M

    def _compute_full_matrix(self):
        """Compute the product of all layer matrices (clean, no noise)."""
        M = self._build_layer_matrix(0, noise_sigma=0.0)
        for i in range(1, self._n_layers):
            M = M @ self._build_layer_matrix(i, noise_sigma=0.0)
        return M

    def forward(self, x, noise_sigma=0.0, n_mc_trials=0):
        """
        Args:
            x: real input [batch, N]
            noise_sigma: phase noise std in radians (0 = no noise)
            n_mc_trials: if > 0 and noise_sigma > 0, run this many
                         independent MC trials in parallel.
                         Returns [n_mc_trials, batch, N] intensities.
        Returns:
            intensities: [batch, N] or [n_mc_trials, batch, N] after photodetection
        """
        state = x.to(torch.complex64)

        if n_mc_trials > 0 and noise_sigma > 0:
            # Vectorized MC: state becomes [n_trials, batch, N]
            state = state.unsqueeze(0).expand(n_mc_trials, -1, -1)
            for layer_idx in range(self._n_layers):
                M = self._build_layer_matrix(layer_idx, noise_sigma, n_mc_trials)
                # state: [T, B, N], M: [T, N, N] -> bmm
                state = torch.bmm(state, M)
            return photodetect(state)

        if noise_sigma == 0.0 and not self.training:
            # Clean eval: use cached full-mesh matrix
            if self._cached_matrix is None:
                self._cached_matrix = self._compute_full_matrix()
            state = state @ self._cached_matrix
        else:
            # Training or single noisy pass: per-layer matmul
            self._cached_matrix = None
            for layer_idx in range(self._n_layers):
                M = self._build_layer_matrix(layer_idx, noise_sigma)
                state = state @ M

        return photodetect(state)

    def get_mzi_count(self):
        return self.n_mzis

    def get_optical_depth(self):
        return self.optical_depth

    def get_total_crossings(self):
        return self._total_crossings

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
                 classifier_hidden=(64,), nonlinearity='photodetect',
                 crossing_loss_dB=0.0):
        super().__init__()
        self.N = N
        self.topology_name = topology
        self.n_photonic_layers = n_photonic_layers
        self.nonlinearity = nonlinearity

        # Photonic layers
        self.meshes = nn.ModuleList([
            PhotonicMesh(N, topology, loss_per_mzi_dB, crossing_loss_dB)
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

    def forward(self, x, noise_sigma=0.0, n_mc_trials=0):
        """Forward pass through photonic + electronic layers."""
        state = x
        has_trials = (n_mc_trials > 0 and noise_sigma > 0)

        for i, mesh in enumerate(self.meshes):
            if self.nonlinearity == 'modReLU' and i > 0:
                # Re-encode as complex for next photonic layer
                state_complex = state.to(torch.complex64)
                state_complex = modReLU(state_complex, self.modrelu_biases[i-1])
                state = mesh(state_complex.real, noise_sigma, n_mc_trials)
            else:
                state = mesh(state, noise_sigma, n_mc_trials)

        if has_trials:
            # state: [T, B, N] -> reshape for classifier -> reshape back
            T, B, N = state.shape
            state = self.classifier(state.reshape(T * B, N))
            return state.reshape(T, B, -1)
        else:
            return self.classifier(state)

    def forward_with_noise(self, x, sigma, n_mc_trials=0):
        return self.forward(x, noise_sigma=sigma, n_mc_trials=n_mc_trials)

    def get_mzi_count(self):
        return sum(m.get_mzi_count() for m in self.meshes)

    def get_optical_depth(self):
        return sum(m.get_optical_depth() for m in self.meshes)

    def get_phase_params(self):
        all_thetas = torch.cat([m.thetas for m in self.meshes])
        all_phis = torch.cat([m.phis for m in self.meshes])
        return all_thetas, all_phis
