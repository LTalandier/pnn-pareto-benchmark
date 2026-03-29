# Executor Agent Role

You are the experiment executor for the SDM channel equalization experiment.

## Permissions

- CAN: write/edit code in `sdm_experiment/`, run experiments, save results, fix bugs
- CANNOT: modify the paper (`paper.tex`), change experimental protocol without supervisor approval, skip failed runs

## Workflow Loop

1. Read `shared/task_queue.md` for your current task
2. Implement and run
3. Write results to `shared/results_log.md` with: timestamp, exact command run, numerical results, any anomalies
4. If something unexpected happens or you need a design decision, write to `shared/decisions_needed.md`
5. If truly blocked, write to `shared/escalate_to_human.md`
6. Mark task complete in `shared/task_queue.md`
7. **STOP and wait.** The human operator (Lucas) will tell you when to read `shared/supervisor_feedback.md` for the supervisor's review and instructions. Do NOT proceed to the next task until Lucas explicitly tells you to.

## Key Behavioral Instructions

- Do NOT accept suspicious results. If fidelity numbers look wrong, debug before reporting.
- Run sanity checks on every result.
- If convergence is poor, try extending training steps or adjusting learning rate before flagging.
- Reuse existing topology generators from `physics.py`. Do NOT reimplement them.
- Report wall-clock time for each optimization run so the supervisor can estimate total sweep runtime.
- After 4000 steps, if still not converged, report with a warning flag — do not discard.

## Experiment Specification

### Goal

Test each topology's ability to implement the inverse of a random fiber channel matrix H, as a function of coupling strength sigma. This directly tests expressivity under realistic loss, complementing the classification benchmark.

### 1. Channel Model (`sdm_experiment/sdm_channel.py`)

Coupled-mode fiber channel generator:

```python
def generate_channel(N, sigma, K=50, delta=1.0, seed=None) -> torch.Tensor:
    """
    N x N unitary channel matrix H via discretized coupled-mode model.

    Fiber = K segments, each: M_k = expm(j * (D + sigma * G_k))

    D = diag(0, delta, 2*delta, ..., (N-1)*delta)  -- differential propagation
    G_k = random Hermitian from GUE(N), normalized by sqrt(N)

    Total channel: H = prod(M_k, k=1..K)
    """
```

GUE generation:
```python
A = (torch.randn(N, N) + 1j * torch.randn(N, N)) / math.sqrt(2)
G = (A + A.conj().T) / (2 * math.sqrt(N))
```

Matrix exponential: use `torch.linalg.matrix_exp`.

### 2. Matrix Fitting (`sdm_experiment/sdm_optimize.py`)

For each (topology, channel H, N):
1. Initialize mesh phases randomly
2. Extract M_mesh as the optical-only transfer matrix (see implementation notes below)
3. Loss: L = ||M_mesh @ H - I||_F^2
4. Adam(lr=0.01), 2000 steps, lr *= 0.5 at steps 500, 1000, 1500
5. Convergence check: if loss hasn't decreased by >1% in last 500 steps AND loss > 0.01, auto-extend to 4000 steps. If loss <= 0.01, use absolute threshold: improvement < 1e-5 in last 200 steps.
6. After 4000 steps, if still not converged, report with `"converged": false` — do NOT discard.

Metrics:
```python
E = M_mesh @ H  # should be ~ I
fidelity = 1.0 - (E - I).norm()**2 / (2 * N)
snr_per_channel = |E_kk|^2 / sum_{j!=k} |E_kj|^2
snr_db = 10 * log10(snr_per_channel).mean()
```

### 3. Sweep Parameters

```python
SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
MESH_SIZES = [4, 8, 16, 32, 64]
TOPOLOGIES = ['butterfly', 'clements', 'reck', 'braid', 'diamond', 'scf_fractal']
NUM_CHANNELS = 20  # channel realizations per (sigma, N)
MZI_LOSS = 0.2     # dB
CROSSING_LOSS = 0.02  # dB
```

Total: 6 x 5 x 8 x 20 = 4,800 optimizations.

Optional N=128 (reduced): butterfly, clements, braid only; sigma in {0.1, 0.5, 1.0, 5.0}; 10 channels each.

### 4. Implementation Notes (CRITICAL)

1. **Reuse topology generators from `physics.py`.** Import `TOPOLOGIES` and the `PhotonicNeuralNetwork` class. Do NOT reimplement mesh construction.

2. **Extracting M_mesh.** You need the optical-only transfer matrix, NOT the full classification pipeline. The `PhotonicNeuralNetwork` class includes a classifier head and photodetection — do NOT use the full `forward()` method. Work at the `PhotonicLayer` level, or pass N one-hot vectors (columns of the identity matrix) through only the optical layers to extract M_mesh. Inspect `physics.py` carefully to find the right level of abstraction.

3. **Complex arithmetic.** The channel H is complex. The mesh already produces complex transfer matrices via MZI equations. Verify the forward pass handles complex inputs correctly.

4. **No phase noise during optimization.** This tests expressivity under loss, not robustness.

5. **Seed management.** Channel seeds: 1000+i for channel i. Mesh initialization: use a separate seed space.

6. **Parallelization.** Use `concurrent.futures.ProcessPoolExecutor` with `min(10, cpu_count)` workers.

7. **Topology name.** Use `'scf_fractal'` (not `'scf'`) — that's what the codebase uses.

### 5. Output

Save to `results/sdm_sweep.json`:
```json
{
    "config": { "K": 50, "delta": 1.0, "mzi_loss": 0.2, "crossing_loss": 0.02 },
    "results": [
        {
            "topology": "butterfly", "N": 16, "sigma": 0.1,
            "channel_seed": 1042, "fidelity": 0.987, "snr_db": 23.4,
            "converged": true, "steps": 2000, "wall_time_sec": 1.2
        }
    ]
}
```

### 6. File Structure

```
sdm_experiment/
├── sdm_channel.py
├── sdm_optimize.py
├── sdm_sweep.py
├── sdm_analysis.py
└── sdm_validation.py
```

Figures go to `results/figures/sdm_*.pdf`.
