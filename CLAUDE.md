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

## SDM Experiment (Active)

A dual-agent workflow is set up in `shared/`. Two Claude Code sessions coordinate through shared files:

- **Executor** (`shared/executor_role.md`): implements code, runs experiments, reports results
- **Supervisor** (`shared/supervisor_role.md`): evaluates results, assigns tasks, writes paper sections

Coordination files:
- `shared/task_queue.md` — supervisor assigns, executor reads
- `shared/results_log.md` — executor writes, supervisor evaluates
- `shared/decisions_needed.md` — executor flags questions for supervisor
- `shared/escalate_to_human.md` — either agent escalates to Lucas
