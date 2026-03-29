# Research Roadmap — Beyond the SDM Experiment

This document gives the supervisor agent the full strategic picture. Read this after completing the SDM channel equalization experiment. These are the next research directions, in priority order. Do NOT start any of these without human approval — this is context for planning, not a task queue.

---

## Current state (when you read this)

The paper "MZI mesh topology comparison for photonic neural networks" has been submitted or is about to be submitted to arXiv. It compares six MZI mesh topologies (Clements, Reck, Butterfly/FFT, SCF Fractal, Braid, Diamond) with realistic loss models from N=4 to N=128. The SDM channel equalization experiment has been added to address the universality criticism.

Key findings established:
- Depth is the binding constraint on PNN scalability
- Butterfly/FFT dominates the Pareto front at all tested scales
- Path-length asymmetry is an advantage under insertion loss (Diamond vs Reck)
- Matrix fidelity does not predict classification performance (Braid collapse contradicts Marchesin et al.)
- Crossing crosstalk is negligible through N=128
- SDM experiment quantifies when universality matters (the expressivity-depth crossover point)

Codebase: https://github.com/LTalandier/pnn-pareto-benchmark
- Differentiable forward pass for 6 topologies with insertion loss + crossing loss
- Monte Carlo phase noise evaluation
- Multi-seed training with Adam optimizer
- Pareto front computation
- SDM coupled-mode channel model (added recently)

---

## Research direction 1: Automated topology discovery (HIGHEST PRIORITY)

### The question
Is butterfly actually optimal, or just the best topology humans have thought of? Can we discover a topology that combines O(log N) depth with sub-quadratic crossing count and greater expressivity than butterfly?

### Why this matters
- Nobody has done loss-aware topology search. ADEPT/ADEPT-Z (Gu et al.) search over supermeshes but without realistic loss models — their results don't transfer to the fabrication regime.
- The paper already identifies the "holy grail" as a topology combining logarithmic depth with sub-quadratic crossings (Section 4.7, Limitation 5).
- If the search discovers a novel topology, it's a major publication and validates PhotonForge as a platform.
- If the search proves butterfly is Pareto-optimal, that's also publishable — it tells the field to stop looking.

### Search space design (the hard problem)
Three possible parameterizations, from constrained to open:

1. **Structural primitives:** compose known building blocks — butterfly stages, Clements blocks, skip connections, braid stages, truncated universals. Search over compositions. Manageable space, interpretable results, but may miss truly novel structures.

2. **Graph-level:** topology as a directed graph where nodes are MZIs and edges are waveguide connections. Depth and crossing count computable from graph structure. Search over graphs subject to physical constraints (bounded crossing count, connected paths). Large space but physically grounded.

3. **Continuous relaxation (ADEPT-style):** supermesh with learnable pruning weights on every possible connection. Add our loss models to the differentiable forward pass. Gradient-based search. Largest space but may converge to known topologies and results are hard to interpret.

Recommendation: start with structural primitives (option 1) for tractability. If nothing novel emerges, escalate to graph-level (option 2).

### Key design heuristics from the paper
Encode these in the search — they constrain the space and prevent wasting compute:
- Depth is what kills performance. Prioritize candidates with depth < N/2.
- Path balancing doesn't help (Braid result). Don't bias search toward balanced topologies.
- Crossing count grows quadratically for butterfly. Favor candidates with sub-quadratic crossings.
- Over-parameterization doesn't help (Diamond result). More MZIs ≠ better.
- Path-length variance can be beneficial (Reck result). Allow asymmetric structures.

### Compute requirements
- Each candidate topology evaluation: ~5 min at N=64 (4 seeds, Vowel dataset)
- Screening phase: 500 candidates × 5 min = ~42 hours sequential, ~3 hours on 16 cores
- Validation phase: top 10 candidates × full benchmark (4 datasets, noise sweep) = ~1 day
- Total: 2–3 days on a 32-core cloud machine (Hetzner CCX53, ~€15)

### What success looks like
- A topology with depth O(log N) or O(√N), crossing count sub-quadratic, and classification accuracy ≥ butterfly
- OR: proof that no topology in the search space beats butterfly on the Pareto front (negative result)
- Either outcome is a standalone paper

---

## Research direction 2: Multi-layer benchmarking (QUICK WIN)

### The question
Does butterfly still dominate when you stack 2–3 optical layers with nonlinearities between them? Can many shallow layers replace one deep layer?

### Why this matters
- The paper's Limitation 2 explicitly calls this out as an open question
- Multi-layer architectures are where the field is heading (Xue et al. 2024, Ashtiani et al. 2026, Zhou et al. 2025 — all already cited in the paper)
- Quick to implement: the new code is just the multi-layer forward pass (~100 lines of PyTorch)

### Experiment design
- Topologies: Butterfly, Clements, Reck (the three that survived at N=64)
- Layer counts: 1 (baseline), 2, 3
- Nonlinearity: electronic ReLU between layers (optical → photodetect → ReLU → re-encode → next layer)
- Mesh sizes: N=16, 32, 64
- Datasets: Vowel, Fashion-MNIST
- Evaluation: same protocol as paper (clean accuracy, noise robustness, Pareto front)

### Key hypothesis
If many shallow butterfly layers outperform one deep Clements layer at equal total MZI count, the depth thesis extends to multi-layer: you should always prefer shallow-and-wide over deep-and-narrow, even with nonlinearities available.

### Compute requirements
- 3 topologies × 3 layer counts × 2 mesh sizes × 2 datasets × 4 seeds = 144 runs
- ~12 hours sequential, ~2 hours on 8 cores
- Runs on local laptop (no cloud needed)

### What success looks like
- A clear answer to "does the topology ranking change in multi-layer?"
- If butterfly still dominates: the depth thesis is universal, not architecture-specific
- If the ranking changes: new design principles for multi-layer PNNs
- Standalone short paper or letter

---

## Research direction 3: PhotonForge platform (ENGINEERING, NOT RESEARCH)

### What it is
Package the research codebase into an autonomous photonic topology optimization platform. A user inputs foundry parameters, the agent explores topologies overnight, produces a Pareto analysis report.

### Architecture (from existing plan)
```
photonforge/
├── core/           # Simulation engine (existing code, refactored)
│   ├── topologies/ # Plugin interface — one file per topology
│   ├── physics.py  # MZI model, loss, noise, photodetection
│   ├── network.py  # PNN forward pass
│   ├── train.py    # Training loop
│   └── evaluate.py # Metrics and Pareto
├── foundry/        # Foundry parameter profiles (YAML)
├── agent/          # Autoresearch loop
│   ├── program.md  # Agent instructions (encodes research heuristics)
│   ├── runner.py   # Experiment orchestration with parallel dispatch
│   └── journal.py  # Experiment logging
├── datasets/       # Data preparation
└── results/        # Auto-populated
```

### Key engineering tasks
1. Refactor codebase into plugin interface (topology as plugin)
2. Foundry profile YAML loader
3. Parallel experiment runner (multiprocessing.Pool or joblib)
4. Agent loop: runner.py reads program.md, plans experiments, logs results
5. Auto-generated summary report

### When to build this
After research directions 1 and 2 produce results. The platform is the productization of the research, not a prerequisite for it. Don't build infrastructure before you have findings worth automating.

### Commercialization path (long-term, human decision)
- Open-source release with papers as credibility
- Consulting: run PhotonForge on client foundry parameters
- SaaS: hosted version for design houses
- Licensing to EDA vendors
- None of these decisions should be made by the agent

---

## Research direction 4: Beyond MZIs (FUTURE, NOT NOW)

### What it is
Extend the framework to include non-MZI components: microring resonators (MRRs), phase-change materials (PCMs), multimode interference couplers (MMIs). Enable hybrid topology search across component types.

### Why not now
- Each component requires a new differentiable physics model
- MRR model alone (resonance lineshape, wavelength detuning, thermal crosstalk) is a research project
- The MZI-only search needs to work first to validate the methodology
- This is v2.0 of PhotonForge, not v1.0

### Component abstraction to design now (even if not implemented)
```python
class Component(ABC):
    def forward(self, input_field) -> output_field: ...
    def loss_db(self) -> float: ...
    def n_params(self) -> int: ...
```
If the topology plugin interface uses this abstraction from the start, adding MRRs later doesn't require a refactor.

---

## Sequencing summary

```
NOW:        SDM experiment (in progress)
            ↓
NEXT:       Multi-layer benchmarking (1–2 weeks, laptop)
            Topology search design (1 week thinking, then 2–3 weeks compute)
            ↓
THEN:       Papers 2 and 3 from these results
            PhotonForge refactor and open-source release
            ↓
LATER:      Hybrid component search (MRR, PCM)
            PhotonForge commercialization
```

## Rules for the supervisor agent

- Do NOT start a new research direction without explicit human approval
- When planning experiments for direction 1 or 2, write the plan to shared/task_queue.md and wait for human review before the executor begins
- If results from any direction suggest a pivot (e.g., topology search finds something that changes the multi-layer hypothesis), flag it in shared/escalate_to_human.md
- These directions may be reordered or cancelled based on Dr. Argyris's feedback, funding opportunities, or strategic decisions that are not yours to make
- Your job is to be aware of the full picture so you can plan intelligently within whichever direction is active
