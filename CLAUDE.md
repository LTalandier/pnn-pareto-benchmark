# PNN Pareto Benchmark

Read `program.md` first — it contains your full instructions.

## File Roles
- `physics.py` — MZI primitives + 6 topology generators. FIXED.
- `prepare.py` — Data loading (vowel, fmnist, iris). FIXED.
- `evaluate.py` — Training + multi-noise evaluation + logging. FIXED.
- `analyze.py` — Pareto front + robustness + scaling plots. FIXED.
- `train.py` — Experiment configuration. **YOU MODIFY THIS.**
- `program.md` — Your full instructions and strategy guide.
- `results/` — Experiment log, figures, and final summary.

## Quick Reference
```bash
python train.py          # Run one experiment
python analyze.py        # Generate all plots from logged data
cat results/experiment_log.jsonl | python -m json.tool | tail -20
git log --oneline -10    # Recent experiments
```

## Rules
- Only modify train.py
- One configuration change per experiment
- git commit after each successful run
- Do the systematic sweep FIRST before any creative experiments
- Write results/summary.md when stopping
