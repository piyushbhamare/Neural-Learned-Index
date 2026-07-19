# Neural Learned Index (NLI)

A single-file C++17 learned index implementation benchmarked against B-Tree, ALEX, PGM, and RMI on real-world SOSD datasets. Includes a one-command Python runner that handles dependency setup, compilation, benchmarking, and figure generation.

---

## Project Layout

```
learned_index_complete/
├── sosd_benchmark_final.cpp   # Complete NLI C++ implementation + all baselines
├── nli_master.py              # One-script runner: setup → compile → run → figures
├── CMakeLists.txt             # Optional CMake build (C++17, -O3 -march=native)
├── cleanup_folder.bat         # Windows utility to remove superseded old files
├── third_party/               # Auto-downloaded by nli_master.py
│   ├── alex/                  # Microsoft ALEX header-only library
│   └── pgm/                   # PGM-index header-only library
├── sosd_data/                 # SOSD binary dataset files (place here before running)
│   ├── books_200M_uint64
│   ├── fb_200M_uint64
│   └── wiki_ts_200M_uint64
├── results/                   # Generated benchmark CSVs
│   ├── benchmark_results.csv        # Read latency — all algorithms × datasets × scales
│   ├── write_results.csv            # Insert latency — all algorithms
│   ├── mixed_workload_results.csv   # Read/write ratio sweep (10/90, 50/50, 90/10)
│   ├── ablation_results.csv         # NLI component ablation
│   ├── drift_results.csv            # Drift detection (F1, precision, recall per scenario)
│   ├── drift_ensemble_ablation.csv  # Per-detector metrics: EWMA / PSI / KS / AE
│   ├── drift_overhead_results.csv   # Overhead % per detector and combined ensemble
│   ├── scalability_results.csv      # Latency at 100K → 1M → 10M → 50M keys
│   └── training_log.csv             # Hyperparams, convergence, build time per dataset
└── figures/                   # Publication-quality PNG figures (300 DPI)
    ├── fig1_latency_comparison.png
    ├── fig2_speedup_heatmap.png
    ├── fig3_drift_detection.png
    ├── fig4_ensemble_ablation.png
    ├── fig5_ablation.png
    ├── fig6_scalability.png
    ├── fig7_training_log.png
    └── fig8_drift_overhead.png
```

---

## Architecture

NLI is a two-component learned index: a piecewise linear CDF model corrected by a small neural network, with an ensemble drift detector and a buffered insert layer.

### NLILinModel — 16-segment piecewise OLS

The data range is split into 16 equal-rank segments. Each segment gets an independent ordinary least-squares linear fit, reducing within-segment MAE from ~191 positions (single global fit on Books 1M) to ~12 positions. Segment lookup is a 4-level branchless binary decision (4 integer ops, no branches).

### TinyMLP — 1 → 16 → 1 residual corrector

A small two-layer network trained to predict the residual between the linear model's position estimate and the true rank.

| Parameter | Value |
|---|---|
| Architecture | 1 → 16 → 1 (ReLU hidden) |
| Parameters | 33 (16 + 16 + 1) |
| Init | He initialisation, seed 42 |
| Optimiser | SGD with cosine learning rate decay |
| Base LR | 5×10⁻⁴ |
| Epochs | 75 (early stop: loss < 1e-8 or < 0.02 % gain for 5 epochs) |
| Training sample | 15,000 points per dataset |
| AVX2/FMA | Auto-vectorised: 2 YMM passes per forward pass |

### CAAB — Confidence-Aware Adaptive Bounds

The search window around the predicted position is sized proportionally to the calibration MAE, rather than using a fixed error bound. This keeps the window tight on easy datasets and widens it automatically under distribution shift.

### EIDD-S — Ensemble Incremental Drift Detection (sampled)

Four detectors run in parallel, sampled every 128 queries to amortise overhead. A drift alarm fires on majority vote (≥ 2 / 4):

| Detector | Method |
|---|---|
| EWMA | Z-score of exponentially weighted mean error vs stable baseline (α = 0.005, z > 2.5) |
| PSI | Population Stability Index across 16 equal-width buckets (threshold > 0.20) |
| KS | Two-sample Kolmogorov-Smirnov test on rolling 2000-query window (α = 0.01) |
| AE | TinyAE autoencoder (16→4→16, 84 params) reconstruction error on rolling error windows |

### ETIR — Efficient Targeted Incremental Repair

On confirmed drift: warm-start repair — flush the BIB → refit OLS segments → fine-tune MLP (15 epochs, 5–8 effective) → recalibrate CAAB. Completes in 5–15 ms.

### BIB — Buffered Insert Buffer

Inserts land in an unsorted buffer (up to 4,096 keys). Buffer is lazily merged and the model is retrained once the buffer fills. This gives O(1) amortised insert cost without structural rebuilds on every write.

---

## Requirements

- **C++ compiler**: GCC 9+, Clang 10+, or MSVC 2019+ with C++17 support
- **Python 3** with `numpy`, `pandas`, `matplotlib`, `scipy` (auto-installed by `nli_master.py`)
- **CMake ≥ 3.15** (optional — only needed if using the CMake build path)

---

## Dataset Setup

Download the three SOSD binary datasets and place them in `sosd_data/`:

```
sosd_data/
├── books_200M_uint64       # Book price data — smooth, nearly sorted
├── fb_200M_uint64          # Facebook user IDs — clustered, non-uniform
└── wiki_ts_200M_uint64     # Wikipedia edit timestamps — quasi-periodic
```

**Format**: raw binary — 8-byte little-endian uint64 key count, followed by N × 8-byte sorted uint64 keys.

Download from the [SOSD repository](https://github.com/learnedsystems/SOSD) or its Harvard Dataverse mirror.

---

## Running

### Recommended: nli_master.py (does everything)

```bash
# Full interactive mode
python nli_master.py

# Quick run — 100K keys only, skips figures
python nli_master.py --quick

# Regenerate figures from existing result CSVs only
python nli_master.py --figs
```

`nli_master.py` runs these phases automatically:

1. **Deps** — checks and installs Python packages
2. **Headers** — downloads official ALEX and PGM headers from GitHub into `third_party/`
3. **Compile** — compiles `sosd_benchmark_final.cpp` with `-O3 -march=native -DUSE_REAL_ALEX -DUSE_REAL_PGM`
4. **Benchmark** — interactive menu to select key scale (100K / 1M / both / full up to 50M)
5. **Figures** — generates all 8 publication-quality figures into `figures/`

### Manual compile (without nli_master.py)

```bash
# With official ALEX + PGM headers (after running nli_master.py at least once)
g++ -O3 -std=c++17 -march=native \
    -Ithird_party \
    -DUSE_REAL_ALEX -DUSE_REAL_PGM -DALEX_USE_LZCNT=0 \
    -o nli_benchmark sosd_benchmark_final.cpp

# Run (SOSD data in ./sosd_data/, results go to ./results/)
./nli_benchmark

# Without official headers (uses faithful reference implementations)
g++ -O3 -std=c++17 -march=native -o nli_benchmark sosd_benchmark_final.cpp
```

### CMake (optional)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

---

## Output CSVs

| File | Content |
|---|---|
| `benchmark_results.csv` | Read latency (mean ns), insert latency (mean ns), build time (ms), memory (KB) for all algorithms × datasets × scales |
| `write_results.csv` | Insert latency per algorithm |
| `mixed_workload_results.csv` | Read/write ratio sweep: 10/90, 50/50, 90/10 at 1M keys |
| `ablation_results.csv` | NLI with each component removed, per dataset and scale |
| `drift_results.csv` | Drift detection F1, precision, recall, FPR per drift type and window size |
| `drift_ensemble_ablation.csv` | Each of the 4 detectors (EWMA / PSI / KS / AE) evaluated independently |
| `drift_overhead_results.csv` | Per-detector query overhead as % of baseline latency |
| `scalability_results.csv` | NLI vs B-Tree latency from 100K to 50M keys |
| `training_log.csv` | Per-dataset: segments, epochs, LR, sample size, final loss, MAE, std, build time |

---

## Key Results

### Read latency — 1M keys, mean (ns)

| Algorithm | Books | Facebook | WikiTS |
|---|---|---|---|
| B-Tree | 408.9 | — | — |
| ALEX | 44.6 | — | — |
| PGM | 79.5 | — | — |
| RMI | 79.0 | — | — |
| **NLI** | competitive across all three datasets | | |

### Mixed workload — NLI at 1M keys, Books

| Read ratio | Read latency (ns) | Insert latency (ns) |
|---|---|---|
| 10 % reads / 90 % writes | 264.4 | 1,704.7 |
| 50 % / 50 % | 192.8 | 3,256.0 |
| 90 % reads / 10 % writes | 171.6 | 1.3 |

### Training (Books, 1M keys)

| Param | Value |
|---|---|
| Hidden units | 16 |
| Segments | 16 |
| Epochs | 75 |
| LR | 5×10⁻⁴ |
| Final MAE | 52.3 positions |
| Build time | 32.8 ms |
| Seed | 42 |

---

## Figures

| Figure | Content |
|---|---|
| `fig1_latency_comparison.png` | Mean read latency — all algorithms, all datasets |
| `fig2_speedup_heatmap.png` | NLI speedup vs baselines across datasets and scales |
| `fig3_drift_detection.png` | Drift detection performance across scenarios |
| `fig4_ensemble_ablation.png` | Per-detector precision / recall / F1 / FPR |
| `fig5_ablation.png` | NLI ablation study — component contribution |
| `fig6_scalability.png` | Read latency from 100K to 50M keys |
| `fig7_training_log.png` | Training convergence curves |
| `fig8_drift_overhead.png` | Drift detection overhead % per detector |

---

## Compile Flags Note

Always build with these flags to use the official baseline implementations:

```
-DUSE_REAL_ALEX -DUSE_REAL_PGM -DALEX_USE_LZCNT=0
```

Without these flags, the code falls back to faithful reference implementations of ALEX and PGM built into `sosd_benchmark_final.cpp`, which can be used for reproducibility if the official headers are unavailable.

---

## Third-Party Libraries

| Library | Source | License |
|---|---|---|
| ALEX | [microsoft/ALEX](https://github.com/microsoft/ALEX) | Apache 2.0 |
| PGM-index | [gvinciguerra/PGM-index](https://github.com/gvinciguerra/PGM-index) | Apache 2.0 |
