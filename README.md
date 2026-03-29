# PNN Topology Pareto Benchmark

Systematic Pareto comparison of MZI mesh topologies for photonic neural networks, evaluated across accuracy, hardware cost, noise robustness, and waveguide crossing loss from N=4 to N=128.

**Paper**: "MZI mesh topology comparison for photonic neural networks: Pareto analysis with crossing loss from N=4 to N=128" (Talandier, 2026).

## How this was made

This project was built as an **autoresearch** experiment, inspired by Andrej Karpathy's concept of [LLM-driven research workflows](https://github.com/karpathy/autoresearch). A human researcher (Lucas Talandier) defined the research question, physics model, and evaluation protocol in a structured prompt file (`pnn-pareto-prompt.md`), then an AI agent (Claude, via [Claude Code](https://docs.anthropic.com/en/docs/claude-code)) autonomously generated the codebase, ran 300+ experiments, produced figures, and wrote the LaTeX paper.

The human's role was to: define the research scope, verify topology implementations against the literature, make judgment calls on what to include, direct experimental priorities, and review the paper for correctness and tone.

Every experiment is logged in `results/experiment_log.jsonl` with full configuration, random seed, and per-noise-level accuracy. The git history shows the complete research trajectory.

## Project structure

```
physics.py              # MZI primitives + 6 topology generators + crossing loss model
prepare.py              # Data loading (Vowel, MNIST, Fashion-MNIST, CIFAR-10)
evaluate.py             # Training + multi-noise Monte Carlo evaluation (vectorized)
generate_figures.py     # Generate all paper figures (7 PNG/PDF)
train.py                # Experiment configuration (modified per run)
paper.tex               # Full LaTeX manuscript (article class, arXiv-ready)
paper.pdf               # Compiled paper
results/
  experiment_log.jsonl  # Classification experiments (145 runs)
  sdm_sweep.json        # SDM channel equalization sweep (4,800 jobs)
  figures/              # All generated plots
sdm_experiment/
  sdm_channel.py        # Coupled-mode SDM fiber channel generator
  sdm_optimize.py       # Matrix fitting optimization (L-BFGS / Adam)
  sdm_sweep.py          # Full parameter sweep runner
  sdm_analysis.py       # Plotting and analysis
  sdm_validation.py     # Channel model sanity checks
shared/                 # Dual-agent coordination files
```

## Topologies

| Topology | MZIs | Depth | Universal | Crossings | Reference |
|----------|------|-------|-----------|-----------|-----------|
| Clements | N(N-1)/2 | N | Yes | 0 | Clements et al., Optica 2016 |
| Reck | N(N-1)/2 | 2N-3 | Yes | 0 | Reck et al., PRL 1994 |
| Butterfly/FFT | N/2 log2(N) | log2(N) | No | O(N^2) | Tian et al., Nanophotonics 2022 |
| Braid | N(N-1)/2 | N-1 | Yes | O(N^2) | Marchesin et al., Opt. Express 2025 |
| Diamond | ~(N-1)^2 | 2N-3 | Yes | 0 | Shokraneh et al., Opt. Express 2020 |
| SCF Fractal | N(N-1)/2 | N-1 | Yes | O(N^2) | Basani et al., Nanophotonics 2023 |

## Key findings

- **Butterfly dominates the Pareto front at all tested scales** (N=4 to 128), achieving +3.3 to +4.8 pp higher accuracy than Clements at N=64 across all four datasets, with up to 2.8x better noise robustness and 100x lower hardware cost.
- **Optical depth is the binding constraint**, not MZI count, universality, or path balancing. At N=128, Clements collapses to 26.3% while butterfly retains 76.6%.
- **Path balancing does not compensate for depth**: Braid (perfectly balanced, depth N-1) collapses to 11.4% at N=128, worse than Clements, contradicting fidelity-based predictions from Marchesin et al. 2025.
- **Over-parameterization does not compensate for depth**: Diamond (~(N-1)^2 MZIs, depth 2N-3) collapses at N=64 (28.3%) while Reck (same depth class) survives at 81.5%, showing that path-length asymmetry is an advantage under insertion loss.
- **Crossing loss is measurable but secondary** at current technology (0.02 dB/crossing): 0.7 pp at N=16, 3.2 pp at N=128 for butterfly.
- **Noise-aware training cannot close the depth gap**: butterfly *without* noise-aware training outperforms Clements *with* it on both accuracy and robustness.
- **SDM channel equalization confirms the loss-driven advantage**: on lossless meshes, universal topologies achieve near-perfect channel inversion at all coupling strengths; under realistic insertion loss, butterfly's raw fidelity exceeds Clements's below a scale-dependent crossover (sigma* ~ 0.5 at N=32).

## Computational setup

### Classification experiments
- **Hardware**: local workstation (10-core / 20-thread AMD CPU, no GPU)
- **Optimizer**: Adam (lr=0.005, 100 epochs per run)
- **Scale**: 300+ experiments across 6 topologies, N=4 to 128, 4 datasets, 4 seeds each
- **Total compute**: ~30 hours cumulative

### SDM channel equalization
- **Hardware**: Hetzner CX53 cloud server (16 vCPUs, 32 GB RAM)
- **Optimizer**: L-BFGS (200 outer steps, 20 line-search iterations each). Adam was initially used but found to systematically underconverge on universal topologies at high coupling strength (e.g., Clements N=32 sigma=5.0: Adam fidelity 0.894 vs L-BFGS 0.995).
- **Scale**: 4,800 jobs (6 topologies x 5 mesh sizes x 8 coupling strengths x 20 channels), each with lossy + lossless optimization
- **Total compute**: ~30 minutes wall time with 16 parallel workers

### Dual-agent workflow
The SDM experiment used a dual-agent workflow with two Claude Code sessions coordinating through shared files (`shared/`):
- **Executor agent**: implemented code, ran experiments, reported results
- **Supervisor agent**: reviewed results against physics expectations, made protocol decisions, wrote paper sections
- **Human (Lucas Talandier)**: approved all decisions, managed hardware, relayed messages between agents

This workflow is documented in `shared/executor_role.md` and `shared/supervisor_role.md`.

## Usage

```bash
pip install torch torchvision numpy matplotlib scikit-learn

# Classification experiments
python train.py              # Run one experiment (edit config in train.py)
python generate_figures.py   # Generate all paper figures

# SDM channel equalization
python sdm_experiment/sdm_validation.py   # Validate channel model
python sdm_experiment/sdm_sweep.py --N 4 8 16 32 64 --workers 16 --optimizer lbfgs
```

## Adding a topology

Implement a function in `physics.py` that returns a list of layers, where each layer is a list of `(port_i, port_j)` tuples representing MZIs that execute in parallel:

```python
def my_topology(N):
    layers = []
    # ... build layers ...
    return layers

TOPOLOGIES['my_topology'] = my_topology
```

Then configure `train.py` with `TOPOLOGY = 'my_topology'` and run.

## License

MIT
