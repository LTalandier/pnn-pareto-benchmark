# Supervisor Feedback

The supervisor writes task reviews, approvals, and instructions here.
The executor reads this file when told to by the human operator.

Format: timestamp, task reviewed, verdict (approved/rejected/revision needed), instructions.

---

## Task 1 Review — 2026-03-27

**Verdict: APPROVED**

All validation checks pass. Channel model is correct:
- Unitarity errors ~1e-7 (float64 internal was the right call)
- Weak coupling clearly nearly-diagonal (0.0095 off-diag)
- Strong coupling shows Haar-like spread (ratio 1.66 is acceptable for K=50)
- Intensity plots confirm expected progression

No issues found in the code. Proceed to Task 2.

---

## Task 2 Review — 2026-03-27

**Verdict: APPROVED with protocol addition**

### Results assessment

Code and results are solid:
- `get_mesh_matrix()` correctly works at the `PhotonicMesh` level (not `PhotonicNeuralNetwork`), handles the right-multiply → left-multiply transpose — this is correct
- Convergence curves are clean: smooth descent, clear plateaus, no instability
- Clements converges in 2000 steps, butterfly auto-extends to 4000 and plateaus higher — expected
- Butterfly variance at σ=0.5 (one channel at 0.848, others 0.67-0.70) is physically sensible: some channels align with butterfly's restricted connectivity
- Wall-clock 2-5s at N=8 is good

### Decision 1 resolution: insertion loss floor

**Decision: use BOTH raw and normalized fidelity.**

The primary metric stays raw fidelity at 0.2 dB/MZI (consistent with the classification experiments, tells the practical story). Additionally, for each (topology, N) combination, run a **lossless baseline** (loss=0 dB/MZI, crossing_loss=0 dB) to isolate the expressivity component.

**Implementation for the sweep:**

For each (topology, N, sigma, channel) run TWO optimizations:
1. **Lossy** (0.2 dB/MZI, 0.02 dB/crossing) — the primary metric
2. **Lossless** (0 dB/MZI, 0 dB/crossing) — the expressivity baseline

Report both:
- `fidelity_raw`: fidelity from the lossy run (primary)
- `fidelity_lossless`: fidelity from the lossless run (expressivity ceiling)

This doubles the optimization count (4,800 → 9,600), but at 2-5s each for N=8 it should be manageable. If wall time becomes a concern at larger N, we can reduce to 10 channels instead of 20 for the lossless runs.

Update `results/sdm_sweep.json` format to include both metrics per entry:
```json
{
    "topology": "butterfly", "N": 16, "sigma": 0.1,
    "channel_seed": 1042,
    "fidelity_raw": 0.987, "snr_db_raw": 23.4,
    "fidelity_lossless": 0.995, "snr_db_lossless": 28.1,
    "converged_raw": true, "converged_lossless": true,
    "steps_raw": 2000, "steps_lossless": 2000,
    "wall_time_sec": 4.2
}
```

### Convergence criterion fix

The auto-extend falsely triggered on Clements σ=0.01 because loss plateaued at the insertion-loss floor (~0.18 > 0.01 threshold). Fix: raise the absolute threshold to 0.05, or better, detect the floor by checking if loss is stable (std of last 200 steps < 1e-4) regardless of its value. This matters more now that we have lossless runs (where the 0.01 threshold is appropriate) and lossy runs (where the floor can be much higher).

### Parallelization note

Set `torch.set_num_threads(2)` at the top of each worker process to avoid thread oversubscription (10 workers × default 20 PyTorch threads = 200 threads on 20 hardware threads). Keep 10 workers as specified.

### Proceed to Task 3

Update `sdm_optimize.py` with the dual-metric approach and convergence fix, then proceed to Task 3 (full sweep N <= 32).

---

## Task 3 Review — 2026-03-28

**Verdict: APPROVED**

Results are physically sound and the crossover story is clear:
- N=16: butterfly wins raw fidelity at σ<0.2, Clements overtakes at σ≈0.2
- N=32: crossover shifts to σ≈0.5 — butterfly's loss advantage extends further at larger N
- Lossless fidelity confirms Clements beats butterfly at ALL sigmas (no expressivity crossover — the raw crossover is purely loss-driven)
- Diamond raw fidelity is σ-invariant (over-parameterized, loss-floor-limited) — expected
- Braid/SCF track each other perfectly — expected (same MZI counts)

The dual-metric approach works exactly as intended.

### Task 4: N=64 — REVISED PROTOCOL

Analysis of Task 3 shows the auto-extend is wasting time: **83% of N=32 lossy runs extended to 4000 steps for nothing** (sitting at the insertion loss floor). Restarts also show no benefit — results are consistent across seeds, not rescued by restarts.

**Revised N=64 sweep — apply ALL of these changes:**

1. **3 topologies only:** butterfly, clements, reck (the three that tell the crossover story)
2. **10 channels** instead of 20 (sufficient statistics — N=32 std was small)
3. **1 restart** instead of 3 (N=32 showed restarts don't rescue anything at this scale)
4. **No auto-extend for lossy runs.** Cap lossy runs at 2000 steps. The insertion loss floor is reached well before 2000 steps — extending to 4000 is pure waste. Keep auto-extend for lossless runs only.
5. All 8 sigma levels as before

Total: 3 × 8 × 10 × 2 = 480 jobs, each ~1-4 min. Estimated time: **1-2 hours** with 10 workers.

Save results appended to `results/sdm_sweep.json`. Report the same table format as Task 3.

**BUT FIRST — run the optimizer benchmark below before starting the sweep.**

### Task 4a: L-BFGS vs Adam benchmark (run BEFORE the N=64 sweep)

The current Adam optimizer may be slower than necessary. L-BFGS uses curvature information and typically converges in far fewer iterations for smooth problems like phase optimization.

**Benchmark protocol:**
1. Pick 5 (topology, sigma, channel) combinations from the N=32 data: clements σ=0.5, clements σ=5.0, butterfly σ=0.5, butterfly σ=5.0, reck σ=5.0. Use channel seed 1000 for all.
2. For each, run the optimization twice:
   - **Adam**: current settings (lr=0.01, milestones [500,1000,1500], 2000 steps). Both lossy and lossless.
   - **L-BFGS**: `torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=10)`. Run for 100-200 outer steps (each step does up to 20 line-search evals internally). Both lossy and lossless.
3. Report for each: final fidelity, wall-clock time, number of steps/evals.
4. The fidelity should match within ±0.005. If L-BFGS is faster AND matches fidelity, use it for the N=64 sweep. If it's worse on fidelity, stick with Adam.

**Important L-BFGS notes:**
- L-BFGS requires a closure function: `def closure(): optimizer.zero_grad(); loss = ...; loss.backward(); return loss`
- Don't use a learning rate scheduler with L-BFGS — it handles step sizes internally via line search
- If L-BFGS is unstable (NaN loss), try reducing lr to 0.1

Report results to `shared/results_log.md`, then proceed with whichever optimizer won for the N=64 sweep.

---

## Task 4a Review — 2026-03-28

**Verdict: L-BFGS wins decisively. FULL RE-RUN REQUIRED.**

### Key finding

Adam was **systematically underconverged** on universal topologies at high σ:
- Clements N=32 σ=5.0 lossless: Adam 0.894 vs L-BFGS 0.995 — a **0.10 gap**
- All existing lossless fidelities for Clements/Reck at σ≥0.5 are understated
- Butterfly unaffected (L-BFGS finds the same ceiling, just 7-9x faster)

This means the Task 3 lossless results are not publishable as-is. The true expressivity gap between universal and non-universal topologies is even larger than reported.

### Revised plan: FULL RE-RUN with L-BFGS

**Scrap the existing `results/sdm_sweep.json` and re-run everything from scratch using L-BFGS.**

This will run on a Hetzner CX53 (16 vCPUs). Updated protocol:

**Optimizer:** L-BFGS for all runs. Settings:
- `torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=10)`
- 200 outer steps max
- Convergence: stop early if loss improvement < 1e-8 over 10 steps
- No learning rate scheduler (L-BFGS handles step sizes via line search)
- If NaN encountered, restart that single run with lr=0.1

**Sweep parameters — NO nerfing:**
- N = 4, 8, 16, 32, 64
- All 6 topologies: butterfly, clements, reck, braid, diamond, scf_fractal
- All 8 sigma levels: 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0
- 20 channels per (topology, N, sigma)
- 1 restart (L-BFGS doesn't need restarts)
- Dual metrics: lossy (0.2 dB/MZI, 0.02 dB/crossing) + lossless (0/0)

**Total: 6 × 5 × 8 × 20 = 4,800 jobs × 2 loss conditions = 9,600 optimizations.**

**Workers:** `min(16, cpu_count)` — use all available vCPUs. Set `torch.set_num_threads(1)` per worker.

**Output:** Save to `results/sdm_sweep.json` (fresh file, delete the old one). Same JSON format as before.

**Reporting:** When complete, write results to `shared/results_log.md` with the same table format (topology × sigma, mean±std, for both fidelity_raw and fidelity_lossless) for each N.

**Optional N=128 (if time permits):** butterfly, clements, braid only. sigma in {0.1, 0.5, 1.0, 5.0}. 10 channels.

Proceed.

---

## Task 4 (full L-BFGS sweep) Review — 2026-03-29

**Verdict: APPROVED. Excellent data.**

4,800/4,800 jobs completed. L-BFGS fixed the underconvergence — results are now publishable. Key findings from the supervisor's analysis:

1. **Crossover shifts with N as predicted**: σ*≈0.05 (N=16), σ*≈0.5 (N=32), σ*≈1.0 (N=64)
2. **Butterfly at N=64 σ=0.01: 0.954 raw vs Clements 0.708** — a 24.6pp gap from insertion loss
3. **Lossless fidelity confirms 100% loss-driven crossover**: Clements achieves 0.979 at N=64 σ=5.0, butterfly collapses to 0.362
4. **Reck degrades at N=64 σ=5.0** (0.804 lossless vs Clements 0.979) — optimization difficulty, not a bug
5. **Diamond: perfect lossless (1.000), worst raw (0.559)** — depth kills it despite over-parameterization

Paper sections have been updated with final numbers. Now generate the figures.

### Task 5: Analysis and figures

Generate the following figures and save to `results/figures/`:

1. **`sdm_crossover.pdf`** — Primary figure. Fidelity vs σ (log x-axis), one panel per N (N=4,8,16,32,64). Each panel has 6 curves (one per topology). Use **solid lines** for raw fidelity, **dashed lines** for lossless fidelity. Use the same color scheme as the paper's existing figures if possible (check `generate_figures.py` for the color map). Mark the crossover point σ* with a vertical dashed gray line in each panel where it exists.

2. **`sdm_crossover_point.pdf`** — σ* vs N plot. X-axis: N (log scale). Y-axis: crossover σ* (log scale). One point per N where a crossover exists (N=16, 32, 64). Fit a trend line if it looks clean.

3. **`sdm_heatmaps.pdf`** — Equalized channel intensity |M_mesh @ H|² at N=16 for butterfly and Clements, at σ=0.01, 0.5, 5.0. 2×3 grid (rows: butterfly, Clements; columns: σ values). This shows visually how well each topology inverts the channel.

4. **`sdm_n64_bar.pdf`** — Bar chart of raw fidelity at N=64 for all 6 topologies at σ=0.01 and σ=5.0 (grouped bars). Clearly shows the ranking inversion.

Save all as PDF. Write a brief description of each figure to `shared/results_log.md`.

Then STOP and wait.
