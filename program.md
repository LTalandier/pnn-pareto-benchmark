# PNN Pareto Benchmark — Agent Instructions

## Mission

Produce a systematic, publication-quality comparison of 6 MZI mesh
topologies for photonic neural networks. The deliverable is a set of
Pareto front plots showing accuracy vs hardware cost vs robustness,
plus a written analysis in results/summary.md.

Target: Optics Express or Nanophotonics paper.

## What You Modify

ONLY `train.py`. Everything else is fixed.

## The Six Topologies

| Topology | MZIs | Depth | Universal? | Key property |
|----------|------|-------|------------|--------------|
| clements | N(N-1)/2 | N | Yes | Standard reference |
| reck | N(N-1)/2 | 2N-3 | Yes | Deeper, less parallel |
| butterfly | N/2·log₂N | log₂N | No | Compact, low depth |
| diamond | ~N(N-1)/2 | ~N | Yes | Balanced path lengths |
| braid | ~N² | ~2N | Yes | Perfectly balanced paths |
| scf_fractal | N(N-1)/2 | ~N·log₂N | Yes | Error scales as O(Nσ⁴) |

## Experiment Protocol

### Phase 1: Systematic Sweep (priority — do this first)

Run every topology at every mesh size on both datasets.
This is the core data for the paper.

**Configurations to sweep:**
- Topologies: all 6
- Mesh sizes: 4, 8, 16, 32 (64 if time permits)
- Datasets: 'vowel', 'fmnist'
- Settings: 1 photonic layer, loss=0.2 dB, photodetect, 100 epochs

That's 6 × 4 × 2 = 48 experiments minimum.

For each experiment:
1. Edit train.py with the configuration
2. Run: `python train.py`
3. If it completes successfully: `git add -A && git commit -m "sweep: <topology> N=<size> <dataset> | clean_acc=<acc>"`
4. If it crashes (e.g. topology requires power-of-2): note in log, skip
5. Move to next configuration

Work through this systematically. Do NOT skip ahead to advanced
experiments before the sweep is complete.

### Phase 2: Generate Plots

After the sweep, generate all plots:
```bash
python analyze.py
```

Examine the Pareto front plots. Identify:
- Which topologies are Pareto-optimal at each noise level?
- Are there crossover points where one topology overtakes another?
- How does scaling (N) affect the topology ranking?

### Phase 3: Targeted Follow-ups

Based on Phase 2 analysis, run targeted experiments:

**Insertion loss sensitivity:**
- Re-run Pareto-optimal topologies at loss = 0.1, 0.3, 0.5 dB
- This tests whether the ranking changes with foundry quality

**Multi-layer architectures:**
- Try 2-layer and 3-layer versions of top topologies
- Use modReLU nonlinearity between layers

**Noise-aware training:**
- Re-run top topologies with NOISE_AWARE_TRAINING=True
- Compare robustness improvement

**Pruning (if time permits):**
- For Clements: reduce N_PHOTONIC_LAYERS or remove MZIs by fixing
  some phases to 0 (passthrough). Report accuracy vs MZI count.

### Phase 4: Analysis and Summary

Write `results/summary.md` containing:

1. **Main finding**: Which topology is Pareto-optimal, under what
   conditions? Is there a single winner or regime-dependent ranking?

2. **Scaling behavior**: How does the topology ranking change from
   N=4 to N=32? At what N does butterfly overtake Clements?

3. **Robustness analysis**: Which topologies degrade most gracefully
   under phase noise? Does the braid's path-balancing actually help?

4. **Insertion loss impact**: Does higher loss change the ranking?
   (This is important for real foundries.)

5. **Design guidelines**: If a chip designer has N=16 ports, σ=0.05
   phase error, and 0.3 dB/MZI loss, which topology should they use?

6. **Limitations**: What simplifications were made? (No crossing loss,
   no wavelength dependence, etc.)

7. **Suggested next steps**: What experiments would strengthen these
   results? (3D FDTD validation, experimental verification, etc.)

Also re-run `python analyze.py` to generate final publication figures.

## Stopping Criteria

**Stop and write summary if ANY of these is true:**
1. Phase 1 sweep is complete AND Phase 2 plots are generated AND
   Phase 3 has at least 10 follow-up experiments.
2. You have run 150 total experiments.
3. A clear, publishable Pareto front has emerged with ≥3 topologies
   showing distinct operating regimes.

## Physical Parameters (use these values)

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss per MZI | 0.2 dB (default), sweep 0.1-0.5 | AIM Photonics PDK |
| Phase noise σ | 0.0 to 0.2 rad | Typical SiPh foundry |
| Phase range | [0, 2π] | Standard thermo-optic |

## What Good Results Look Like

- All 6 topologies × ≥3 mesh sizes × 2 datasets = ≥36 data points
- Clear Pareto front with at least 2-3 topologies on the frontier
- Robustness curves showing distinct degradation profiles
- Scaling plot showing crossover points
- A written analysis with concrete design guidelines
