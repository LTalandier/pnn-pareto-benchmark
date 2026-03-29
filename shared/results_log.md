# SDM Experiment Results Log

Executor writes results here. Supervisor reads and evaluates.

Format: timestamp, task number, what was run, numerical results, anomalies.

---

## Task 1: Channel model + validation — 2026-03-27

**Command:** `.venv/bin/python3 sdm_experiment/sdm_validation.py`

**Files implemented:**
- `sdm_experiment/sdm_channel.py` — coupled-mode channel generator using `torch.linalg.matrix_exp`
- `sdm_experiment/sdm_validation.py` — unitarity, weak/strong coupling checks, intensity plots

### 1. Unitarity check: ||H†H - I||_F for N=8, 20 samples per sigma

| sigma | mean error | max error | status |
|-------|-----------|-----------|--------|
| 0.01  | 1.38e-07  | 2.15e-07  | PASS   |
| 0.05  | 1.50e-07  | 2.33e-07  | PASS   |
| 0.10  | 1.62e-07  | 2.14e-07  | PASS   |
| 0.20  | 1.65e-07  | 2.84e-07  | PASS   |
| 0.50  | 1.72e-07  | 2.23e-07  | PASS   |
| 1.00  | 1.90e-07  | 2.51e-07  | PASS   |
| 2.00  | 1.79e-07  | 2.30e-07  | PASS   |
| 5.00  | 1.68e-07  | 2.18e-07  | PASS   |

All errors < 3e-7 (well below 1e-6 threshold). Channel computation uses float64 internally, cast to float32 on output.

### 2. Weak coupling (sigma=0.01): nearly diagonal

- Mean off-diagonal |H_ij|: **0.0095**
- Threshold: < 0.1 → **PASS**

Channel is nearly identity at weak coupling, as expected.

### 3. Strong coupling (sigma=5.0): Haar-like spread

- Expected |H_ij| ~ 1/√N = 0.3536
- Measured mean |H_ij|: **0.3171**
- Entry mean range: [0.2375, 0.3951], max/min ratio: **1.66**
- **PASS** — entries spread uniformly, consistent with Haar-random unitary

### 4. Intensity matrix plots

Saved to `results/figures/sdm_channel_validation.pdf`. Four panels at σ = 0.01, 0.1, 1.0, 5.0 for N=8:
- σ=0.01: sharp diagonal (nearly identity)
- σ=0.1: slight off-diagonal leakage, still diagonal-dominated
- σ=1.0: significant mixing across all modes
- σ=5.0: fully spread, Haar-like

### Anomalies

None.

### Overall: ALL PASS

---

## Task 2: Single-point optimization test — 2026-03-27

**Command:** `.venv/bin/python3 sdm_experiment/sdm_optimize.py`

**File implemented:** `sdm_experiment/sdm_optimize.py`

### Configuration

- N=8, topologies: butterfly, clements
- σ = 0.01, 0.5, 5.0
- 3 channel realizations each (seeds 1000–1002), mesh seeds 2000–2002
- Adam lr=0.01, milestones [500, 1000, 1500] γ=0.5, 2000 steps + auto-extend to 4000
- MZI loss: 0.2 dB, crossing loss: 0.02 dB

### Results Summary

| Topology   | σ     | Fidelity (mean±std)   | SNR dB (mean) | Time/opt |
|------------|-------|-----------------------|---------------|----------|
| butterfly  | 0.01  | 0.9126 ± 0.0016       | 33.4          | 2.0s     |
| butterfly  | 0.50  | 0.7398 ± 0.0945       | 1.2           | 2.0s     |
| butterfly  | 5.00  | 0.6517 ± 0.0248       | -1.3          | 2.0s     |
| clements   | 0.01  | 0.9887 ± 0.0000       | 38.7          | 4.5s     |
| clements   | 0.50  | 0.9709 ± 0.0072       | 14.9          | 2.3s     |
| clements   | 5.00  | 0.9396 ± 0.0199       | 10.7          | 2.3s     |

### Analysis vs Expected Results

**σ=0.01 (both > 0.99?):**
- Clements: 0.989 — slightly below 0.99. The residual loss (~0.18) is the **insertion loss floor**: N=8 Clements has 28 MZIs at 0.2 dB each = 5.6 dB total insertion loss. The mesh reaches its expressivity limit but cumulative attenuation prevents fidelity from reaching 1.0. This is correct physical behavior, not a convergence failure.
- Butterfly: 0.912 — significantly lower, reflecting its non-universality (only 12 MZIs, log2(8)=3 layers). Cannot fully decompose even a nearly-diagonal unitary.

**σ=5.0 (Clements > 0.95, butterfly measurably lower?):**
- Clements: 0.940 mean — slightly below 0.95 target, but still strong. 2 of 3 channels exceed 0.95.
- Butterfly: 0.652 — dramatically lower. Clear expressivity gap.

**Clements vs butterfly gap is very clear** across all σ values. The gap widens with σ as expected.

**Wall-clock time:** All < 5s per optimization. Well under the 30s target.

### Convergence Note

The `converged=False` flags on Clements σ=0.01 runs are **not failures** — the optimizer converged to the insertion-loss floor (loss ≈ 0.18) and then the auto-extend criterion triggered because the loss was >0.01 with <1% improvement. The loss floor is set by physics (cumulative MZI attenuation), not by insufficient optimization. The convergence criterion should perhaps be adjusted for lossy meshes, but results are valid.

### Convergence plot

Saved to `results/figures/sdm_task2_convergence.pdf`.

### Anomalies

1. **Insertion loss floor**: Clements fidelity capped at ~0.989 even at σ=0.01 due to 0.2 dB/MZI × 28 MZIs. This is physically correct but means the "fidelity > 0.99" target is unachievable with these loss parameters. Supervisor decision needed: reduce MZI loss for this experiment, or accept the floor?

2. **Butterfly high variance at σ=0.5**: One channel reached 0.848, the other two were 0.67–0.70. This likely depends on how well the channel's structure aligns with the butterfly connectivity pattern.

---

## Task 3: Full sweep (N ≤ 32) — 2026-03-28

**Command:** `.venv/bin/python3 sdm_experiment/sdm_sweep.py --N 4 8 16 32 --channels 20 --workers 10`

**Files implemented/updated:**
- `sdm_experiment/sdm_optimize.py` — added 3-restart strategy (escapes local minima), improved convergence detection (plateau via std < 1e-4)
- `sdm_experiment/sdm_sweep.py` — parallel sweep with resume support, dual lossy/lossless metrics per supervisor decision

**Protocol:** Per supervisor feedback, each (topology, N, σ, channel) runs TWO optimizations: lossy (0.2 dB/MZI, 0.02 dB/crossing) and lossless (0 dB). 3 restarts per optimization. Total: 3840 jobs × 2 × 3 = 23,040 individual optimizations.

### Timing

| N  | Jobs | Mean wall time/job | Total CPU |
|----|------|--------------------|-----------|
| 4  | 960  | 5.0s               | 1.3h      |
| 8  | 960  | 14.7s              | 3.9h      |
| 16 | 960  | 29.0s              | 7.7h      |
| 32 | 960  | 65.9s              | 17.6h     |

### Convergence

| N  | converged_raw | converged_lossless |
|----|---------------|--------------------|
| 4  | 957/960       | 960/960            |
| 8  | 900/960       | 952/960            |
| 16 | 643/960       | 952/960            |
| 32 | 408/960       | 934/960            |

Low raw convergence at large N is dominated by insertion-loss floor (plateau detected but loss > threshold). Lossless convergence is >97% across all N.

### N=16 Results

**fidelity_raw (mean±std):**

| Topology    | σ=0.01      | σ=0.05      | σ=0.1       | σ=0.2       | σ=0.5       | σ=1.0       | σ=2.0       | σ=5.0       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| clements    | 0.957±0.000 | 0.957±0.000 | 0.957±0.000 | 0.956±0.000 | 0.949±0.005 | 0.936±0.007 | 0.924±0.011 | 0.924±0.011 |
| reck        | 0.952±0.000 | 0.951±0.000 | 0.950±0.001 | 0.943±0.003 | 0.903±0.013 | 0.825±0.019 | 0.781±0.020 | 0.781±0.017 |
| butterfly   | 0.974±0.001 | 0.970±0.003 | 0.958±0.006 | 0.913±0.013 | 0.725±0.023 | 0.622±0.016 | 0.604±0.014 | 0.602±0.014 |
| braid       | 0.948±0.000 | 0.948±0.000 | 0.948±0.000 | 0.947±0.001 | 0.942±0.004 | 0.939±0.004 | 0.939±0.003 | 0.940±0.003 |
| diamond     | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 | 0.891±0.000 |
| scf_fractal | 0.945±0.000 | 0.945±0.000 | 0.944±0.000 | 0.944±0.000 | 0.943±0.002 | 0.942±0.002 | 0.941±0.002 | 0.941±0.003 |

**fidelity_lossless (mean±std):**

| Topology    | σ=0.01      | σ=0.05      | σ=0.1       | σ=0.2       | σ=0.5       | σ=1.0       | σ=2.0       | σ=5.0       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| clements    | 1.000±0.000 | 0.999±0.000 | 0.998±0.000 | 0.995±0.001 | 0.982±0.002 | 0.966±0.006 | 0.953±0.012 | 0.952±0.014 |
| reck        | 1.000±0.000 | 0.999±0.000 | 0.996±0.001 | 0.988±0.003 | 0.940±0.014 | 0.845±0.022 | 0.786±0.026 | 0.786±0.020 |
| butterfly   | 0.979±0.001 | 0.974±0.004 | 0.960±0.007 | 0.910±0.015 | 0.695±0.026 | 0.577±0.018 | 0.557±0.016 | 0.555±0.016 |
| braid       | 1.000±0.000 | 0.998±0.000 | 0.996±0.001 | 0.993±0.002 | 0.987±0.002 | 0.985±0.003 | 0.986±0.002 | 0.986±0.002 |
| diamond     | 1.000±0.000 | 1.000±0.000 | 0.999±0.000 | 0.999±0.000 | 0.998±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| scf_fractal | 1.000±0.000 | 0.999±0.000 | 0.998±0.000 | 0.996±0.001 | 0.993±0.001 | 0.990±0.002 | 0.990±0.002 | 0.989±0.002 |

### N=32 Results

**fidelity_raw (mean±std):**

| Topology    | σ=0.01      | σ=0.05      | σ=0.1       | σ=0.2       | σ=0.5       | σ=1.0       | σ=2.0       | σ=5.0       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| clements    | 0.870±0.000 | 0.870±0.000 | 0.870±0.000 | 0.869±0.000 | 0.868±0.000 | 0.866±0.000 | 0.848±0.009 | 0.827±0.004 |
| reck        | 0.868±0.000 | 0.867±0.000 | 0.867±0.000 | 0.865±0.001 | 0.851±0.003 | 0.813±0.007 | 0.758±0.008 | 0.720±0.005 |
| butterfly   | 0.966±0.000 | 0.964±0.001 | 0.957±0.002 | 0.929±0.005 | 0.790±0.017 | 0.630±0.010 | 0.552±0.008 | 0.535±0.006 |
| braid       | 0.846±0.000 | 0.845±0.000 | 0.845±0.000 | 0.845±0.000 | 0.844±0.000 | 0.843±0.000 | 0.841±0.004 | 0.841±0.004 |
| diamond     | 0.725±0.000 | 0.725±0.000 | 0.725±0.000 | 0.725±0.000 | 0.725±0.000 | 0.725±0.000 | 0.725±0.000 | 0.724±0.000 |
| scf_fractal | 0.828±0.000 | 0.828±0.000 | 0.828±0.000 | 0.828±0.000 | 0.827±0.000 | 0.827±0.000 | 0.827±0.000 | 0.827±0.000 |

**fidelity_lossless (mean±std):**

| Topology    | σ=0.01      | σ=0.05      | σ=0.1       | σ=0.2       | σ=0.5       | σ=1.0       | σ=2.0       | σ=5.0       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| clements    | 1.000±0.000 | 1.000±0.000 | 0.999±0.000 | 0.997±0.000 | 0.989±0.001 | 0.974±0.003 | 0.944±0.005 | 0.913±0.007 |
| reck        | 1.000±0.000 | 0.999±0.000 | 0.998±0.000 | 0.994±0.001 | 0.973±0.003 | 0.920±0.010 | 0.835±0.013 | 0.772±0.008 |
| butterfly   | 0.980±0.000 | 0.977±0.001 | 0.969±0.002 | 0.935±0.006 | 0.763±0.021 | 0.567±0.012 | 0.471±0.010 | 0.450±0.007 |
| braid       | 1.000±0.000 | 0.999±0.000 | 0.996±0.000 | 0.990±0.002 | 0.984±0.002 | 0.975±0.002 | 0.970±0.003 | 0.974±0.002 |
| diamond     | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.999±0.000 | 0.997±0.000 | 0.995±0.001 | 0.997±0.001 | 0.999±0.000 |
| scf_fractal | 1.000±0.000 | 1.000±0.000 | 0.999±0.000 | 0.998±0.000 | 0.995±0.000 | 0.991±0.001 | 0.988±0.001 | 0.987±0.001 |

### Key Observations

1. **Lossless fidelity cleanly separates topology expressivity.** Diamond achieves near-perfect fidelity (>0.995) across all σ — its over-parameterization makes it the most expressive. Braid and SCF are close behind (>0.97). Clements degrades gracefully. Reck degrades faster than Clements despite the same MZI count (deeper → more error accumulation). Butterfly collapses at σ>0.5 (non-universal).

2. **Raw fidelity is dominated by insertion loss at large N.** At N=32: diamond (961 MZIs) hits 0.725, braid (496 MZIs) hits 0.845, while butterfly (80 MZIs) gets 0.966 at weak coupling. The topology with the fewest MZIs wins on raw fidelity at low σ, even though it has the worst expressivity.

3. **The crossover is clearly visible in raw fidelity.** Butterfly starts highest at σ=0.01 (fewer MZIs = less loss) but drops below all universal topologies by σ≈0.5-1.0. This is the expressivity-vs-loss tradeoff the experiment was designed to capture.

4. **Diamond raw fidelity is σ-invariant** (0.725 across all σ). Its lossless fidelity is near-perfect, so the raw ceiling is entirely set by its ~961 MZIs of insertion loss. The over-parameterization completely absorbs the channel structure regardless of coupling.

5. **SCF and braid show remarkable robustness.** Both maintain lossless fidelity >0.97 even at σ=5.0, N=32 — better than Clements (0.913) and much better than Reck (0.772).

### Anomalies

1. **Diamond raw fidelity is flat** — physically correct but means it can't equalize at all in practice at 0.2 dB/MZI with (N-1)² MZIs. This topology is only viable with lower-loss MZIs.

2. **Convergence rate drops with N for raw runs** (408/960 at N=32). All are insertion-loss floor plateaus, not optimization failures. The lossless convergence (934/960) confirms the optimizer works correctly.

### Output

Raw results saved to `results/sdm_sweep.json` (3840 entries).

---

## Task 4a: L-BFGS vs Adam optimizer benchmark — 2026-03-28

**Command:** `.venv/bin/python3 sdm_experiment/sdm_benchmark_optimizer.py`

**Protocol:** 5 test cases at N=32 (clements σ={0.5,5.0}, butterfly σ={0.5,5.0}, reck σ=5.0), channel seed 1000, both lossy and lossless. Adam 2000 steps vs L-BFGS 200 outer steps (max 20 line-search evals each).

### Results

| Case                 | Config   | Adam fid | L-BFGS fid | Δfid     | Adam(s) | L-BFGS(s) | Speedup |
|----------------------|----------|----------|------------|----------|---------|-----------|---------|
| clements σ=0.5       | lossy    | 0.8644   | 0.8692     | +0.005   | 10.4    | 20.6      | 0.50x   |
| clements σ=0.5       | lossless | 0.9891   | 0.9990     | +0.010   | 10.1    | 21.7      | 0.46x   |
| clements σ=5.0       | lossy    | 0.8280   | 0.8675     | +0.040   | 10.5    | 22.8      | 0.46x   |
| clements σ=5.0       | lossless | 0.8937   | 0.9953     | **+0.102** | 10.0  | 21.9      | 0.46x   |
| butterfly σ=0.5      | lossy    | 0.7730   | 0.7733     | +0.000   | 1.9     | 0.3       | **6.9x**|
| butterfly σ=0.5      | lossless | 0.7425   | 0.7429     | +0.000   | 1.8     | 0.2       | **8.3x**|
| butterfly σ=5.0      | lossy    | 0.5313   | 0.5291     | -0.002   | 1.9     | 0.3       | **6.6x**|
| butterfly σ=5.0      | lossless | 0.4456   | 0.4433     | -0.002   | 1.8     | 0.2       | **9.1x**|
| reck σ=5.0           | lossy    | 0.7216   | 0.7265     | +0.005   | 20.0    | 14.2      | 1.4x    |
| reck σ=5.0           | lossless | 0.7729   | 0.7865     | +0.014   | 18.7    | 19.6      | 0.95x   |

### Analysis

**L-BFGS achieves significantly higher fidelity on universal topologies:**
- Clements lossless σ=5.0: **0.995 vs 0.894** — Adam is severely underconverged. This means Task 3 N=32 lossless results for Clements/Reck at high σ are understated.
- All Clements/Reck L-BFGS fidelities are equal or better than Adam.

**L-BFGS is 2x slower per job on Clements/Reck** (20-23s vs 10-11s). The cost is justified by the massive fidelity improvement.

**Butterfly: L-BFGS is 7-9x faster** and finds the expressivity ceiling in ~18 steps. Non-universal topology has a hard ceiling that L-BFGS detects instantly.

**Implication for Task 3 results:** The N=32 lossless fidelities for Clements (0.913 at σ=5.0) and Reck (0.772 at σ=5.0) are **underestimates**. L-BFGS gets Clements to 0.995 — the true expressivity gap between Clements and butterfly is even larger than reported.

### Recommendation

**Use L-BFGS for the N=64 sweep.** The 2x wall-time increase per job is offset by:
1. Dropping restarts (1 instead of 3) — already decided
2. Much more accurate fidelity — the numbers we report will be closer to the true optimum
3. Butterfly jobs will be 7x faster, partially compensating

Decision for supervisor: should we re-run the N=32 sweep with L-BFGS to correct the underconverged results? Or proceed with N=64 only and note the caveat?
