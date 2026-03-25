# PNN Topology Pareto Benchmark

Systematic Pareto comparison of MZI mesh topologies for photonic neural networks, evaluated across accuracy, hardware cost, and fabrication noise robustness.

**Paper**: "Quantitative Pareto comparison of MZI mesh topologies for photonic neural networks under fabrication impairments" (Talandier, 2026).

## How this was made

This project was built as an **autoresearch** experiment, inspired by Andrej Karpathy's concept of [LLM-driven research workflows](https://github.com/karpathy/autoresearch) and the broader move toward AI agents that can run experiments end-to-end. The idea: a human researcher defines the research question and constraints, then an AI agent handles the implementation, execution, and writing.

Concretely, a human researcher (Lucas Talandier) defined the research question, physics model, and evaluation protocol in a structured prompt file (`pnn-pareto-prompt.md`), then an AI agent (Claude, via [Claude Code](https://docs.anthropic.com/en/docs/claude-code)) autonomously:

1. **Generated the codebase** — physics simulation, data preparation, evaluation protocol, analysis scripts
2. **Ran 102 experiments** — systematic sweep of 4 topologies x 3 mesh sizes x 2 datasets, plus insertion loss sensitivity, noise-aware training, and multi-seed statistical validation
3. **Produced publication figures** — Pareto fronts with error bars, robustness curves, scaling plots
4. **Wrote the LaTeX paper** — full REVTeX4-2 manuscript with tikz topology schematics
5. **Iterated on peer review feedback** — fixed references, added statistical validation, corrected methodology

The human's role was to: define the research scope, verify topology implementations against the literature (catching and removing two incorrect implementations), make judgment calls on what to include, direct the experimental priorities, and review the paper for correctness and tone.

### What worked well

- **Experiment throughput**: 102 experiments including multi-seed validation, run and logged automatically with git commits after each
- **Boilerplate elimination**: LaTeX formatting, figure generation, reference management, data table extraction from logs
- **Iterative refinement**: the agent could incorporate review feedback (fix references, add error bars, update tables) in minutes

### What required human intervention

- **Topology correctness**: the agent implemented Diamond and Braid topologies incorrectly (identical wiring to Clements). These were caught by the human and removed. Correct implementations require reading the original papers carefully — something the agent could not do reliably.
- **GPU vs CPU decision**: the agent initially ran on GPU, which was slower due to kernel launch overhead on small sequential MZI operations. The human identified this and switched to CPU.
- **Statistical validity**: the human flagged that single-run results were insufficient and directed the multi-seed experiments.
- **Reference verification**: 5 of 17 references had incorrect author lists or titles in the initial draft. An independent check was needed.
- **Tone and framing**: AI-written text required editing.

### Reproducibility

Every experiment is logged in `results/experiment_log.jsonl` with full configuration, random seed, and per-noise-level accuracy. The git history shows the complete research trajectory.

## Project structure

```
physics.py          # MZI primitives + 4 topology generators
prepare.py          # Data loading (vowel, Fashion-MNIST, iris)
evaluate.py         # Training + multi-noise Monte Carlo evaluation
analyze.py          # Pareto front + robustness + scaling plots
plot_errorbars.py   # Multi-seed error bar plots
train.py            # Experiment configuration (modified per run)
paper.tex           # Full LaTeX manuscript (REVTeX4-2)
paper.pdf           # Compiled paper
program.md          # Agent instructions
results/
  experiment_log.jsonl   # All 102 experiments
  figures/               # All generated plots
  summary.md             # Results summary
```

## Topologies

| Topology | MZIs | Depth | Universal | Reference |
|----------|------|-------|-----------|-----------|
| Clements | N(N-1)/2 | N | Yes | Clements et al., Optica 2016 |
| Reck | N(N-1)/2 | 2N-3 | Yes | Reck et al., PRL 1994 |
| Butterfly/FFT | N/2 log2(N) | log2(N) | No | Tian et al., Nanophotonics 2022 |
| SCF Fractal | N(N-1)/2 | ~N | Yes | Basani et al., Nanophotonics 2023 |

Diamond (Shokraneh 2020) and Braid (Marchesin 2025) topologies are not included — their correct implementation requires non-trivial wiring patterns that we could not validate without access to the original papers. Contributions welcome.

## Key finding

Butterfly/FFT dominates the Pareto front at all tested scales (N=4 to 32), achieving comparable classification accuracy to universal topologies at 15-40x lower hardware cost with 1.5-2x better noise robustness. The advantage is driven by O(log N) optical depth and grows with insertion loss.

## Usage

```bash
pip install torch torchvision numpy matplotlib scikit-learn
python train.py            # Run one experiment
python analyze.py          # Generate plots from logged data
python plot_errorbars.py   # Generate multi-seed error bar plots
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

## Related work and inspiration

- Andrej Karpathy's [discussion on AI-assisted research](https://github.com/karpathy/autoresearch) and the concept of "vibe coding" applied to scientific workflows
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic) — the agent harness used to run this project


## License

MIT
