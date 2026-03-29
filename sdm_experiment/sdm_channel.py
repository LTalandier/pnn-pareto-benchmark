"""
Coupled-mode SDM fiber channel generator.

Generates N x N unitary channel matrices H via a discretized coupled-mode model:
    Fiber = K segments, each: M_k = expm(j * (D + sigma * G_k))
    D = diag(0, delta, 2*delta, ..., (N-1)*delta)  -- differential propagation
    G_k = random Hermitian from GUE(N), normalized by sqrt(N)
    Total channel: H = prod(M_k, k=1..K)
"""

import math
import torch


def generate_channel(N, sigma, K=50, delta=1.0, seed=None):
    """
    Generate an N x N unitary channel matrix H via discretized coupled-mode model.

    Args:
        N: number of spatial modes
        sigma: coupling strength (0 = no coupling, large = strong mixing)
        K: number of fiber segments
        delta: differential propagation constant spacing
        seed: random seed for reproducibility

    Returns:
        H: [N, N] complex unitary matrix
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    # Differential propagation matrix (diagonal)
    D = torch.diag(torch.arange(N, dtype=torch.float64) * delta)
    D = D.to(torch.complex128)

    # Accumulate product of segment matrices
    H = torch.eye(N, dtype=torch.complex128)

    for _ in range(K):
        # GUE random Hermitian, normalized by sqrt(N)
        A_real = torch.randn(N, N, generator=gen, dtype=torch.float64)
        A_imag = torch.randn(N, N, generator=gen, dtype=torch.float64)
        A = (A_real + 1j * A_imag) / math.sqrt(2)
        G = (A + A.conj().T) / (2 * math.sqrt(N))

        # Segment matrix: expm(j * (D + sigma * G))
        M_k = torch.linalg.matrix_exp(1j * (D + sigma * G.to(torch.complex128)))
        H = H @ M_k

    return H.to(torch.complex64)
