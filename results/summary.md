# PNN Topology Pareto Benchmark — Results Summary

## Overview

300+ experiments across 6 MZI mesh topologies (Clements, Reck, Butterfly/FFT,
Braid, Diamond, SCF Fractal), mesh sizes N=4 to N=128, 4 datasets (Vowel,
MNIST, Fashion-MNIST, CIFAR-10), with per-MZI insertion loss, waveguide
crossing loss, and Monte Carlo phase-noise evaluation.

## 1. Main Finding: Depth is the Binding Constraint

Optical depth, not MZI count, universality, path balancing, or
over-parameterization, is the primary determinant of PNN performance
under realistic fabrication conditions.

**At N=64 (Vowel, 0.2 dB/MZI, 0.02 dB/crossing):**

| Topology    | MZIs   | Depth | Cost      | Acc (%)    | Rob   |
|-------------|--------|-------|-----------|------------|-------|
| Butterfly   |    192 |     6 |     1,152 | 78.0±3.5   | 0.393 |
| Reck        |  2,016 |   125 |   252,000 | 81.5±2.4   | 0.268 |
| Clements    |  2,016 |    64 |   129,024 | 73.9±1.7   | 0.138 |
| Braid       |  2,016 |    63 |   127,008 | 69.3±3.3   | 0.151 |
| Diamond     |  3,938 |   125 |   492,250 | 28.3±3.2   | 0.354 |
| SCF Fractal |  2,016 |    63 |   127,008 | 63.7±4.3   | 0.154 |

**At N=128 (Vowel):**

| Configuration             | Acc (%)     | Cost        |
|---------------------------|-------------|-------------|
| Butterfly (0.02 dB cross) | 76.6±2.1    |       3,136 |
| Butterfly (no cross)      | 79.8±2.1    |       3,136 |
| Clements                  | 26.3±2.9    |   1,040,384 |
| Braid (0.02 dB cross)     | 11.4±1.1    |   1,032,256 |

## 2. Six-Topology Comparison at N=16

All topologies perform comparably at small scale (Lx=0):

| Topology    | Vowel      | F-MNIST    | MNIST      | CIFAR-10   | Rob (Vowel) |
|-------------|------------|------------|------------|------------|-------------|
| Butterfly   | 84.3±1.7   | 81.5±0.2   | 92.7±0.1   | 40.5±0.3   | 0.377       |
| Reck        | 83.6±2.3   | 81.0±0.2   | 92.1±0.1   | 40.3±0.5   | 0.252       |
| Clements    | 82.7±0.9   | 80.6±0.4   | 91.7±0.6   | 39.6±0.3   | 0.208       |
| Braid       | 83.0±1.5   | 80.3±0.2   | 91.6±0.4   | 39.5±0.2   | 0.209       |
| Diamond     | 81.3±0.9   | 79.4±0.3   | 90.4±0.4   | 38.5±0.3   | 0.172       |
| SCF Fractal | 84.7±0.9   | 79.9±0.2   | 91.3±0.4   | 39.3±0.3   | 0.210       |

At N=16, accuracy differences are within 3.4 pp. Robustness differences
are already substantial: butterfly achieves 1.8x the robustness of Clements.

## 3. Braid: Path Balancing Does Not Help

Braid (Marchesin et al. 2025) has perfectly balanced paths — every port
traverses the same number of MZIs and crossings. Despite this:

- **N=16**: tracks Clements within 1 pp on all datasets
- **N=64**: 69.3% — worse than Clements (73.9%) due to crossing loss
- **N=128**: 11.4% — worse than Clements (26.3%), collapsed

This contradicts Marchesin et al.'s fidelity-based evaluation showing
braid outperforms Clements. Matrix fidelity does not predict classification
under realistic loss conditions.

## 4. Diamond: Over-parameterization Does Not Help

Diamond (Shokraneh et al. 2020) has ~(N-1)^2 MZIs (more than Clements/Reck's
N(N-1)/2), zero crossings, and symmetric paths. Despite this:

- **N=16**: 81.3% — lowest of all topologies
- **N=64**: 28.3% — collapsed (depth 2N-3=125 accumulates 25 dB loss)

Crucially, Reck has the same depth class (2N-3) but survives at 81.5%.
The difference: Reck's triangular structure allows routing through shorter
paths (path-length asymmetry is an advantage), while Diamond forces every
port through the full depth.

## 5. Crossing Loss Analysis

Crossing loss (0.02 dB/crossing) is measurable but secondary:

- Butterfly at N=16: -0.7 pp from crossing loss
- Butterfly at N=128: -3.2 pp from crossing loss
- SCF Fractal hit harder due to higher crossing density

Crosstalk validation (2x2 coupling matrices, eps=0.001/-30 dB) confirmed
<2 pp additional shift at all scales through N=128. The attenuation-only
model is adequate.

## 6. Noise-Aware Training

At N=16 on Vowel:

| Configuration           | Acc (%) | Rob   |
|-------------------------|---------|-------|
| Clements standard       | 82.7    | 0.208 |
| Clements noise-aware    | 65.8    | 0.281 |
| SCF standard            | 84.7    | 0.210 |
| SCF noise-aware         | 71.3    | 0.262 |
| Butterfly standard      | 84.3    | 0.377 |

Butterfly *without* noise-aware training outperforms Clements and SCF
*with* it on both accuracy and robustness simultaneously.

## 7. Insertion Loss Sensitivity

At N=16, butterfly drops only 2.3 pp from 0.2 to 0.5 dB/MZI, vs 5+ pp
for depth-N topologies. For lossy foundry processes, butterfly's advantage
grows.

## Experiment Log

- 145 experiments in `experiment_log.jsonl` (original 4-topology sweep)
- 47 experiments in `results_overnight.json` (Tracks A/B1/B2 + extra datasets)
- ~100 additional runs for Lx=0 consistency and noise curve generation
- All figures in `figures/` (regenerated with 6 topologies)
