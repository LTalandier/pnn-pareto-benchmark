# Decisions Needed

Executor flags design questions here. Supervisor resolves them.

Format: timestamp, question, context, executor's recommendation.

---

## Decision 1: Insertion loss floor vs fidelity target — 2026-03-27

**Question:** The spec expects fidelity > 0.99 at σ=0.01 for Clements, but the insertion loss floor (0.2 dB/MZI × 28 MZIs = 5.6 dB at N=8) caps fidelity at ~0.989. At larger N the floor will be even lower. Should we:

(a) Keep MZI loss at 0.2 dB and accept the floor (the experiment still cleanly separates topologies)
(b) Reduce MZI loss (e.g., 0.05 dB) to raise the ceiling
(c) Normalize fidelity by the lossless-mesh ideal to factor out insertion loss

**Context:** The fidelity metric `1 - ||E-I||²/(2N)` penalizes both expressivity gaps AND cumulative attenuation. At N=64 with 0.2 dB/MZI, Clements has 2016 MZIs — the insertion loss would dominate and mask topology differences.

**Executor recommendation:** Option (c) — normalize by the insertion-loss-only baseline. Alternatively, use a fidelity metric that only measures angular error (e.g., compare singular vectors), not amplitude. But this is a significant protocol change, so flagging for supervisor decision.

---

## Decision 2: Re-run N≤32 with L-BFGS? — 2026-03-28

**Question:** Task 4a benchmark shows Adam underconverges on Clements/Reck lossless at high σ (e.g., Clements N=32 σ=5.0: Adam 0.894 vs L-BFGS 0.995). The Task 3 lossless results for these topologies are therefore underestimates.

Options:
(a) Re-run N≤32 lossless sweep with L-BFGS (960 jobs × ~20s = ~5 hours)
(b) Proceed with N=64 using L-BFGS and note the N≤32 caveat in the paper
(c) Re-run only the high-σ portion (σ ≥ 0.5, clements+reck only)

**Executor recommendation:** Option (c) — re-run only the underconverged region. Low σ results are already at fidelity ~1.0, no benefit from better optimizer. This is ~120 jobs, ~40 min.
