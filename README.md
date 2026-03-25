# PNN Topology Pareto Benchmark

Systematic Pareto comparison of MZI mesh topologies for photonic neural networks, evaluated across accuracy, hardware cost, noise robustness, and waveguide crossing loss from N=4 to N=128.

**Paper**: "MZI mesh topology comparison for photonic neural networks: Pareto analysis with crossing loss from N=4 to N=128" (Talandier, 2026).

## How this was made

This project was built as an **autoresearch** experiment, inspired by Andrej Karpathy's concept of [LLM-driven research workflows](https://github.com/karpathy/autoresearch). A human researcher (Lucas Talandier) defined the research question, physics model, and evaluation protocol in a structured prompt file (`pnn-pareto-prompt.md`), then an AI agent (Claude, via [Claude Code](https://docs.anthropic.com/en/docs/claude-code)) autonomously generated the codebase, ran 145+ experiments, produced figures, and wrote the LaTeX paper.

The human's role was to: define the research scope, verify topology implementations against the literature (catching and removing two incorrect implementations), make judgment calls on what to include, direct experimental priorities, and review the paper for correctness and tone.

Every experiment is logged in `results/experiment_log.jsonl` with full configuration, random seed, and per-noise-level accuracy. The git history shows the complete research trajectory.

## Project structure

```
physics.py              # MZI primitives + 4 topology generators + crossing loss model
prepare.py              # Data loading (Vowel, MNIST, Fashion-MNIST, CIFAR-10)
evaluate.py             # Training + multi-noise Monte Carlo evaluation (vectorized)
generate_figures.py     # Generate all paper figures (7 PNG/PDF)
train.py                # Experiment configuration (modified per run)
paper.tex               # Full LaTeX manuscript (article class, arXiv-ready)
paper.pdf               # Compiled paper
results/
  experiment_log.jsonl  # All 145+ experiments
  figures/              # All generated plots
```

## Topologies

| Topology | MZIs | Depth | Universal | Crossings | Reference |
|----------|------|-------|-----------|-----------|-----------|
| Clements | N(N-1)/2 | N | Yes | 0 | Clements et al., Optica 2016 |
| Reck | N(N-1)/2 | 2N-3 | Yes | 0 | Reck et al., PRL 1994 |
| Butterfly/FFT | N/2 log2(N) | log2(N) | No | O(N^2) | Tian et al., Nanophotonics 2022 |
| SCF Fractal | N(N-1)/2 | N-1 | Yes | O(N^2) | Basani et al., Nanophotonics 2023 |

Diamond (Shokraneh 2020) and Braid (Marchesin 2025) topologies are not included -- their correct implementation requires non-trivial wiring patterns that we could not validate without access to the original papers. Contributions welcome.

## Key findings

- **Butterfly dominates the Pareto front at all tested scales** (N=4 to 128), achieving +3.3 to +4.8 pp higher accuracy than Clements at N=64 across all four datasets, with up to 2.8x better noise robustness and 100x lower hardware cost.
- **Optical depth is the binding constraint**, not MZI count or universality. At N=128, Clements collapses to near-random accuracy (26.3%) while butterfly retains 76.6%.
- **Crossing loss is measurable but secondary** at current technology (0.02 dB/crossing): 0.7 pp at N=16, 3.2 pp at N=128 for butterfly.
- **Noise-aware training cannot close the depth gap**: butterfly *without* noise-aware training outperforms Clements *with* it on both accuracy and robustness.

## Usage

```bash
pip install torch torchvision numpy matplotlib scikit-learn
python train.py              # Run one experiment
python generate_figures.py   # Generate all paper figures
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
