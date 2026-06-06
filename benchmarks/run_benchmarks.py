#!/usr/bin/env python3
"""Run all benchmarks and print comparison table."""

import torch
import numpy as np
from gpu_optimizer.benchmarks import SphereBenchmark, RastriginBenchmark
from gpu_optimizer.tsp_solver import TSPSolver
from gpu_optimizer.portfolio import PortfolioOptimizer

HAS_CUDA = torch.cuda.is_available()
N_TRIALS = 1000
REDUCED_TRIALS = 200  # for slow benchmarks


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def fmt_time(t):
    return f"{t:.3f}s"


def run_sphere():
    section("Sphere Function (dim=10)")
    bench = SphereBenchmark(dim=10, n_trials=N_TRIALS, steps=2000)
    cpu_seq = bench.run_sequential()
    cpu_par = bench.run_cpu()
    if HAS_CUDA:
        gpu = bench.run_gpu()

    print(f"  CPU sequential  : {fmt_time(cpu_seq['time']):>10s}  cost={cpu_seq['best_cost']:.6f}")
    print(f"  CPU parallel    : {fmt_time(cpu_par['time']):>10s}  cost={cpu_par['best_cost']:.6f}")
    if HAS_CUDA:
        print(f"  GPU parallel    : {fmt_time(gpu['time']):>10s}  cost={gpu['best_cost']:.6f}")
        print(f"  GPU vs CPU-seq  : {cpu_seq['time'] / gpu['time']:.1f}x speedup")
        print(f"  GPU vs CPU-par  : {cpu_par['time'] / gpu['time']:.1f}x speedup")
    return {"cpu_seq": cpu_seq, "cpu_par": cpu_par, "gpu": gpu if HAS_CUDA else None}


def run_rastrigin():
    section("Rastrigin Function (dim=10)")
    bench = RastriginBenchmark(dim=10, n_trials=N_TRIALS, steps=3000)
    cpu_seq = bench.run_sequential()
    cpu_par = bench.run_cpu()
    if HAS_CUDA:
        gpu = bench.run_gpu()

    print(f"  CPU sequential  : {fmt_time(cpu_seq['time']):>10s}  cost={cpu_seq['best_cost']:.6f}")
    print(f"  CPU parallel    : {fmt_time(cpu_par['time']):>10s}  cost={cpu_par['best_cost']:.6f}")
    if HAS_CUDA:
        print(f"  GPU parallel    : {fmt_time(gpu['time']):>10s}  cost={gpu['best_cost']:.6f}")
        print(f"  GPU vs CPU-seq  : {cpu_seq['time'] / gpu['time']:.1f}x speedup")
    return {"cpu_seq": cpu_seq, "cpu_par": cpu_par, "gpu": gpu if HAS_CUDA else None}


def run_tsp():
    section("TSP (20 cities)")
    np.random.seed(42)
    cities = np.random.rand(20, 2).astype(np.float32) * 100
    solver = TSPSolver(cities, n_trials=REDUCED_TRIALS, steps=2000, seed=42)
    seq = solver.run_sequential()
    par = solver.run()
    print(f"  CPU sequential ({REDUCED_TRIALS} trials): {fmt_time(seq['time']):>10s}  dist={seq['best_distance']:.2f}")
    print(f"  {'GPU' if HAS_CUDA else 'CPU'} parallel ({REDUCED_TRIALS} trials): {fmt_time(par['time']):>10s}  dist={par['best_distance']:.2f}")
    print(f"  Speedup         : {seq['time'] / par['time']:.1f}x")
    return {"seq": seq, "par": par}


def run_portfolio():
    section("Portfolio Optimization (5 assets)")
    returns = np.array([0.12, 0.10, 0.07, 0.03, 0.05], dtype=np.float32)
    cov = np.array([
        [0.040, 0.006, 0.002, -0.001, 0.003],
        [0.006, 0.025, 0.004, -0.002, 0.002],
        [0.002, 0.004, 0.016, 0.001, 0.001],
        [-0.001, -0.002, 0.001, 0.009, 0.000],
        [0.003, 0.002, 0.001, 0.000, 0.010],
    ], dtype=np.float32)
    opt = PortfolioOptimizer(returns, cov, min_return=0.06, n_trials=REDUCED_TRIALS, steps=1500, seed=42)
    seq = opt.run_sequential()
    par = opt.run()
    print(f"  CPU sequential ({REDUCED_TRIALS} trials): {fmt_time(seq['time']):>10s}  var={seq['portfolio_variance']:.6f}")
    print(f"  {'GPU' if HAS_CUDA else 'CPU'} parallel ({REDUCED_TRIALS} trials): {fmt_time(par['time']):>10s}  var={par['portfolio_variance']:.6f}")
    print(f"  Speedup         : {seq['time'] / par['time']:.1f}x")
    return {"seq": seq, "par": par}


if __name__ == "__main__":
    print(f"CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = {}
    results["sphere"] = run_sphere()
    results["rastrigin"] = run_rastrigin()
    results["tsp"] = run_tsp()
    results["portfolio"] = run_portfolio()

    section("Summary")
    print(f"  Device: {'CUDA' if HAS_CUDA else 'CPU-only'}")
    print(f"  Trials per benchmark: {N_TRIALS} (TSP/Portfolio: {REDUCED_TRIALS})")
    for name, data in results.items():
        if "gpu" in data and data["gpu"]:
            print(f"  {name:12s} GPU speedup: {data['cpu_seq']['time'] / data['gpu']['time']:.1f}x")
