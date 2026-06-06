# gpu-optimizer

GPU-accelerated parallel simulated annealing for optimization problems, built on PyTorch.

Runs **N simulated annealing trials simultaneously** as tensor operations — either on CUDA GPU or vectorized on CPU — and returns the best solution found across all trials.

## Installation

```bash
pip install -e .
# or
pip install -e ".[dev]"  # includes pytest
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

## What's Included

| Module | Description |
|---|---|
| `ParallelSA` | Core engine: batch N SA trials as parallel tensor ops |
| `SphereBenchmark` | Baseline quadratic optimization (sphere function) |
| `RastriginBenchmark` | Multi-modal optimization with many local minima |
| `TSPSolver` | Small traveling salesman problems (10–30 cities) |
| `PortfolioOptimizer` | Minimize portfolio variance subject to return constraint |

## Quick Start

```python
import torch
from gpu_optimizer import ParallelSA

# Define a batch objective: (N, D) tensor → (N,) costs
def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=1)

sa = ParallelSA(
    objective=sphere,
    dim=10,
    n_trials=1000,
    bounds=(-5.12, 5.12),
    steps=2000,
    seed=42,
)
result = sa.run()

print(f"Best cost: {result['best_cost']:.6f}")
print(f"Solution:  {result['best_solution']}")
print(f"Device:    {result['device']}")
```

Auto-detects CUDA. Falls back to CPU if no GPU available.

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU, PyTorch 2.12.0+cu130, Python 3.10

All benchmarks use seed=42 for reproducibility.

### Continuous Optimization (1000 trials × 2000 steps)

| Benchmark | CPU Sequential | CPU Parallel (tensor) | GPU (CUDA) | GPU vs Sequential |
|---|---|---|---|---|
| **Sphere** (dim=10) | 17.3s | 0.04s (**46x**) | 0.46s (**4x**) | 4x |
| **Rastrigin** (dim=10) | 26.5s | 0.05s (**55x**) | 0.41s (**65x**) | 65x |

> CPU-parallel (vectorized tensor ops on CPU) is extremely fast for these dense matrix problems.
> GPU shines on Rastrigin where the objective involves `cos` and `2π` operations — compute-bound workloads benefit most from CUDA cores.

### TSP — 15 Cities (100 trials × 1000 steps)

| Mode | Time | Best Distance |
|---|---|---|
| CPU Sequential | 2.31s | 320.53 |
| GPU Parallel | 11.89s | 320.53 |

> TSP uses 2-opt swaps which require per-trial indexing loops — not as tensor-friendly.
> GPU overhead dominates for small instances. CPU-parallel is recommended for TSP.

### Portfolio Optimization — 5 Assets (100 trials × 1000 steps)

| Mode | Time | Variance | Return | GPU Speedup |
|---|---|---|---|---|
| CPU Sequential | 2.59s | 0.003788 | 6.01% | — |
| GPU Parallel | 0.43s | 0.003788 | 6.01% | **6x** |

### Key Takeaways

1. **CPU-vectorized parallel SA** (tensor ops on CPU) gives massive speedups (40–55x) over naive sequential loops for continuous optimization
2. **GPU acceleration** adds further gains on compute-heavy objectives (Rastrigin: 65x vs sequential)
3. **For indexing-heavy problems** (TSP), stick with CPU — GPU transfer overhead isn't worth it
4. **Real-world finance**: Portfolio optimizer finds minimum-variance weights satisfying return constraints in <0.5s on GPU

## API Reference

### `ParallelSA(objective, dim, n_trials, ...)`

| Parameter | Default | Description |
|---|---|---|
| `objective` | required | `f(x: Tensor[N,D]) → Tensor[N]` |
| `dim` | required | Search space dimensionality |
| `n_trials` | 1000 | Number of parallel SA trials |
| `bounds` | (-5.12, 5.12) | Per-dimension bounds |
| `initial_temp` | 100.0 | Starting temperature |
| `final_temp` | 1e-4 | Minimum temperature |
| `cooling_rate` | 0.997 | Multiplicative cooling per step |
| `steps` | 2000 | Number of annealing iterations |
| `device` | auto | `"cuda"` or `"cpu"` |
| `seed` | None | Random seed for reproducibility |

Methods:
- `.run()` — parallel SA (GPU or CPU tensor ops)
- `.run_sequential()` — sequential single-trial-at-a-time on CPU

Returns dict with `best_solution`, `best_cost`, `all_best_costs`, `history`, `device`.

### `TSPSolver(cities, n_trials, ...)`

Input: `(N_cities, 2)` coordinate array. Uses 2-opt swaps.

### `PortfolioOptimizer(returns, cov_matrix, min_return, ...)`

Input: expected returns vector and covariance matrix. Softmax-normalizes weights. Penalizes return shortfall.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

23 tests covering:
- Solution quality (sphere near 0, TSP valid tours, portfolio constraints)
- Output shapes and device reporting
- Reproducibility with seeds
- Bounds enforcement
- Monotonic convergence history
- Sequential vs parallel consistency

## Running Benchmarks

```bash
python3 benchmarks/run_benchmarks.py
```

## License

MIT
