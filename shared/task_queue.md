# Task Queue

Supervisor assigns tasks here. Executor reads and executes.

Format: task number, status, description, completion criteria.

---

## Task 1: Channel model + validation

**Status:** COMPLETE (2026-03-27)

Implement `sdm_experiment/sdm_channel.py` and `sdm_experiment/sdm_validation.py`.

Run validation:
1. Unitarity check: ||H†H - I||_F < 1e-6 for N=8, all sigma levels, 20 samples each
2. At sigma=0.01: mean off-diagonal magnitude < 0.1 (nearly diagonal)
3. At sigma=5.0: verify H looks Haar-like (spread across all entries)
4. Plot example |H|^2 intensity matrices at sigma = 0.01, 0.1, 1.0, 5.0 for N=8

**Completion:** write validation results to `shared/results_log.md`. STOP and wait for supervisor approval before Task 2.

---

## Task 2: Single-point optimization test

**Status:** COMPLETE (2026-03-27)

Implement `sdm_experiment/sdm_optimize.py`.

Run single optimizations at N=8 for butterfly and Clements at sigma = 0.01, 0.5, 5.0 (3 channel realizations each = 18 total optimizations).

Report for each:
- Final fidelity and SNR
- Convergence curve (loss vs step)
- Wall-clock time per optimization

Expected results:
- sigma=0.01: both topologies fidelity > 0.99
- sigma=5.0: Clements fidelity > 0.95, butterfly measurably lower
- Each optimization < 30 seconds at N=8

**Completion:** write results to `shared/results_log.md`. STOP and wait for supervisor approval before Task 3.

---

## Task 3: Full sweep (N <= 32)

**Status:** COMPLETE (2026-03-28)

Implement `sdm_experiment/sdm_sweep.py`.

Run sweep: N in {4, 8, 16, 32}, all 6 topologies, all 8 sigma levels, 20 channels each.
Total: 6 x 4 x 8 x 20 = 3,840 optimizations.

Save raw results to `results/sdm_sweep.json`. Report summary statistics (mean +/- std fidelity per topology/N/sigma) to `shared/results_log.md`.

**Completion:** write summary to results_log. STOP and wait.

---

## Task 4: N=64 sweep

**Status:** pending (blocked on Task 3 approval)

Extend sweep to N=64, all 6 topologies, all 8 sigma levels, 20 channels.
Append to `results/sdm_sweep.json`.

This is the key scale where the crossover should be most visible.

**Completion:** write summary to results_log. STOP and wait.

---

## Task 5: Analysis and figures

**Status:** pending (blocked on Task 4 approval)

Implement `sdm_experiment/sdm_analysis.py`. Generate:

1. **Crossover plot** (primary figure): fidelity vs sigma (log scale), one curve per topology, separate panel per N. Use same color scheme as existing paper figures.
2. **Crossover point vs N**: extract sigma* where Clements fidelity first exceeds butterfly. Plot sigma* vs N.
3. **Equalized channel heatmaps**: |M_mesh @ H|^2 at N=16, for weak/moderate/strong coupling x butterfly/Clements/Reck.
4. **Summary table**: selected (sigma, N) combinations, mean +/- std fidelity and SNR.

Save figures to `results/figures/sdm_*.pdf`.

**Completion:** write figure descriptions to results_log. STOP and wait.

---

## Task 6 (optional): N=128 confirmation

**Status:** pending (blocked on Task 5 approval)

Reduced sweep: butterfly, clements, braid only. sigma in {0.1, 0.5, 1.0, 5.0}. 10 channels each.

**Completion:** append to results and results_log.
