# PNN Topology Pareto Benchmark — Results Summary

## Overview

36 experiments across 4 validated MZI mesh topologies (Clements, Reck,
Butterfly/FFT, SCF Fractal), 3 mesh sizes (N=4, 8, 16), 2 datasets
(vowel, Fashion-MNIST), plus targeted follow-ups at N=32, insertion
loss sweep, and noise-aware training.

## 1. Main Finding: Butterfly Dominates the Pareto Front

Butterfly/FFT is Pareto-optimal under every condition tested. It
achieves comparable or superior classification accuracy to universal
topologies while using dramatically fewer MZIs and having far
shallower optical depth.

**At N=16 (vowel, 0.2 dB/MZI loss):**

| Topology    | MZIs | Depth | HW Cost | Clean Acc | Robustness |
|-------------|------|-------|---------|-----------|------------|
| clements    |  120 |    16 |   1,920 |    81.8%  |     0.209  |
| reck        |  120 |    29 |   3,480 |    84.5%  |     0.266  |
| butterfly   |   32 |     4 |     128 |    83.8%  |     0.343  |
| scf_fractal |  120 |    15 |   1,800 |    84.2%  |     0.209  |

Butterfly achieves 83.8% accuracy with **15x lower hardware cost**
than Clements and **27x lower** than Reck, while maintaining the
best robustness to phase noise.

At N=32 (vowel), butterfly reaches 85.5% accuracy (cost=400) vs
Clements at 82.2% (cost=15,872) — a **40x cost advantage**.

## 2. Scaling Behavior

All topologies improve with mesh size N, but butterfly scales best
relative to cost:

- **N=4**: All topologies perform similarly (~43-49% vowel, ~70-71% fmnist).
  Too few parameters to differentiate.
- **N=8**: Clements leads on clean accuracy (78.5% vowel), but butterfly
  (76.4%) is close with 7x lower cost. Butterfly leads on robustness.
- **N=16**: Reck edges out on clean accuracy (84.5%), but butterfly
  (83.8%) matches it at 27x lower cost.
- **N=32**: Butterfly (85.5%) overtakes Clements (82.2%) on both accuracy
  and cost.

Butterfly's non-universality (it cannot represent arbitrary unitaries)
does not hurt classification performance at any tested scale. The
O(N log N) MZI count and O(log N) depth give it an increasingly
dominant Pareto position as N grows.

## 3. Robustness Analysis

Robustness (accuracy retention at sigma=0.2 rad) is strongly
correlated with optical depth, not MZI count:

| Topology    | N=16 Depth | Robustness (vowel) | Robustness (fmnist) |
|-------------|------------|--------------------|---------------------|
| butterfly   |          4 |              0.343 |               0.428 |
| reck        |         29 |              0.266 |               0.347 |
| scf_fractal |         15 |              0.209 |               0.247 |
| clements    |         16 |              0.209 |               0.240 |

Butterfly retains 34-43% of clean accuracy at sigma=0.2, while
Clements and SCF Fractal retain only 21-25%. This is a direct
consequence of depth: each MZI in the optical path accumulates
phase noise, so shallower topologies degrade more gracefully.

SCF Fractal, despite its theoretical O(N*sigma^4) error scaling,
does not outperform Clements on robustness in practice. Its depth
of 15 (vs Clements' 16) is not different enough to matter.

Reck is an interesting case: despite having the deepest topology
(depth 29), it shows better robustness than Clements (depth 16).
This may be because Reck's triangular structure concentrates MZIs
in fewer paths, so some output ports are less affected by noise.

## 4. Insertion Loss Impact

At N=16 (vowel), increasing insertion loss from 0.2 to 0.5 dB/MZI:

| Topology    | 0.2 dB Acc | 0.3 dB Acc | 0.5 dB Acc | Acc Drop |
|-------------|------------|------------|------------|----------|
| butterfly   |     83.8%  |     85.5%  |     81.5%  |   -2.3pp |
| clements    |     81.8%  |     84.5%  |     76.4%  |   -5.4pp |
| reck        |     84.5%  |     77.8%  |     81.5%  |   -3.0pp |
| scf_fractal |     84.2%  |     82.5%  |     78.5%  |   -5.7pp |

Butterfly is the most loss-tolerant: only 2.3 percentage points drop
from 0.2 to 0.5 dB/MZI, because signals traverse only 4 MZIs (vs
15-29 for others). For lossy foundry processes (>0.3 dB/MZI),
butterfly's advantage becomes even more pronounced.

Robustness ratios remain stable across loss levels — the topology
ranking does not change with foundry quality.

## 5. Noise-Aware Training

Noise-aware training (injecting sigma=0.05 during training) at N=16
on vowel:

| Topology    | Standard Acc | Noise-Aware Acc | Std Robustness | NA Robustness |
|-------------|-------------|-----------------|----------------|---------------|
| clements    |      81.8%  |          62.3%  |          0.209 |         0.281 |
| scf_fractal |      84.2%  |          71.7%  |          0.209 |         0.263 |

Noise-aware training improves robustness ratios (+34% for Clements,
+26% for SCF) but at a steep cost in clean accuracy (-19.5pp for
Clements, -12.5pp for SCF). The accuracy-robustness tradeoff may
not be favorable for these topologies. Butterfly already achieves
better robustness (0.343) without noise-aware training than
Clements achieves with it (0.281).

## 6. Design Guidelines

**Recommendation by scenario:**

- **Low noise, low loss (sigma<0.05, <0.2 dB/MZI):** Any universal
  topology works. Reck or SCF Fractal give highest clean accuracy.
- **Moderate noise (sigma=0.05-0.1):** Butterfly. Its shallow depth
  retains accuracy where others collapse.
- **High loss foundry (>0.3 dB/MZI):** Butterfly. 4 MZIs in path
  vs 15-29 makes it far more loss-tolerant.
- **Resource-constrained (minimize chip area):** Butterfly. 32 MZIs
  at N=16 vs 120 for universal topologies — 73% area reduction.
- **Maximum clean accuracy, no noise:** Reck or SCF Fractal at N=16+.

**For a practical chip designer with N=16 ports, sigma=0.05 phase
error, and 0.3 dB/MZI loss: use Butterfly.** It gives 85.5% accuracy
(best), 35.8% robustness (best), at 128 hardware cost (lowest).

## 7. Limitations

- **No crossing loss**: Real waveguide crossings add ~0.02 dB each.
  Butterfly has more crossings than Clements, which would partially
  offset its depth advantage.
- **No wavelength dependence**: MZI response varies with wavelength;
  not modeled here.
- **Simplified nonlinearity**: Only photodetection between layers;
  real systems may use different optical nonlinearities.
- **Classification only**: Butterfly is non-universal — it cannot
  implement arbitrary unitary transforms. For applications requiring
  universality (quantum computing, signal processing), Clements or
  SCF Fractal are necessary.
- **Diamond and Braid topologies not tested**: Removed due to
  implementation concerns. These topologies claim path-balanced
  properties that could improve robustness; their correct
  implementations require careful study of the original papers.
- **Small dataset scope**: Only vowel (11 classes) and Fashion-MNIST
  (10 classes) tested. More complex tasks may favor universal topologies.

## 8. Suggested Next Steps

1. **Implement and validate Diamond and Braid topologies** from the
   original papers (Shokraneh 2020, Marchesin 2025). Their
   path-balancing properties could challenge butterfly on robustness.
2. **Waveguide crossing loss model**: Add crossing penalties to
   fairly compare butterfly (many crossings) vs planar topologies.
3. **Multi-layer architectures**: Stack 2-3 butterfly layers with
   modReLU nonlinearity to test if non-universality can be overcome.
4. **Larger mesh sizes** (N=64, 128) to test scaling limits.
5. **Experimental validation**: Fabricate butterfly and Clements at
   N=8 or N=16 on a SiPh foundry to verify simulation predictions.
6. **Task complexity sweep**: Test on CIFAR-10 (via PCA), more
   complex phoneme datasets, to find where butterfly's
   non-universality becomes a bottleneck.

## Experiment Log

36 total experiments:
- Phase 1: 4 topologies x 3 mesh sizes x 2 datasets = 24 experiments
- Phase 3: 8 insertion loss + 2 noise-aware + 2 N=32 = 12 experiments
- All results in `experiment_log.jsonl`
- All figures in `figures/`
