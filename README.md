# PNN Topology Pareto Benchmark

Systematic comparison of 6 MZI mesh topologies for photonic neural
networks, evaluated across accuracy, hardware cost, and fabrication
noise robustness.

## Topologies
- **Clements** — rectangular, universal, the standard
- **Reck** — triangular, universal, deeper
- **Butterfly/FFT** — O(N log N) compact, non-universal
- **Diamond** — balanced insertion loss paths
- **Braid** — perfectly balanced passive component count
- **SCF Fractal** — recursive CS decomposition, superior error scaling

## Metrics
- Classification accuracy (clean and under phase noise)
- MZI count (hardware cost)
- Optical depth (insertion loss accumulation)
- Robustness (accuracy retention under noise)

## Datasets
- Deterding vowel classification (standard PNN benchmark)
- Fashion-MNIST with PCA reduction

## Usage
```bash
pip install torch torchvision numpy matplotlib scikit-learn
python train.py       # Run configured experiment
python analyze.py     # Generate Pareto front plots
```

## References
- Clements et al., Optica 3, 1460 (2016)
- Reck et al., PRL 73, 58 (1994)
- Tian et al., Nanophotonics 11 (2022)
- Shokraneh et al., Optics Express 28, 23495 (2020)
- Marchesin et al., Optics Express 33 (2025)
- Basani et al., Nanophotonics 12, 5 (2023)
