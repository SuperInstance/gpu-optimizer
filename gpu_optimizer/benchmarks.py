"""Benchmark functions: Sphere and Rastrigin."""

from __future__ import annotations

import torch
import numpy as np
import time
from gpu_optimizer.parallel_sa import ParallelSA


class SphereBenchmark:
    """Minimise the sphere (quadratic) function: f(x) = Σ x_i².

    Global minimum at origin with value 0.
    """

    def __init__(
        self,
        dim: int = 10,
        n_trials: int = 1000,
        steps: int = 2000,
        bounds: tuple[float, float] = (-5.12, 5.12),
    ):
        self.dim = dim
        self.n_trials = n_trials
        self.steps = steps
        self.bounds = bounds

    @staticmethod
    def objective(x: torch.Tensor) -> torch.Tensor:
        """Compute sphere function for batch of points.

        Parameters
        ----------
        x : Tensor, shape ``(N, D)``

        Returns
        -------
        Tensor, shape ``(N,)``
        """
        return (x ** 2).sum(dim=1)

    def run_gpu(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Sphere GPU"
        return result

    def run_cpu(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            device="cpu",
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Sphere CPU-parallel"
        return result

    def run_sequential(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            device="cpu",
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run_sequential()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Sphere CPU-sequential"
        return result


class RastriginBenchmark:
    """Minimise the Rastrigin function: f(x) = A·D + Σ[x_i² − A·cos(2πx_i)].

    Highly multi-modal with many local minima.  Global minimum at origin = 0.
    """

    A = 10

    def __init__(
        self,
        dim: int = 10,
        n_trials: int = 1000,
        steps: int = 3000,
        bounds: tuple[float, float] = (-5.12, 5.12),
    ):
        self.dim = dim
        self.n_trials = n_trials
        self.steps = steps
        self.bounds = bounds

    def objective(self, x: torch.Tensor) -> torch.Tensor:
        """Rastrigin function."""
        return self.A * x.shape[1] + (
            x ** 2 - self.A * torch.cos(2 * torch.pi * x)
        ).sum(dim=1)

    def run_gpu(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            initial_temp=150.0,
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Rastrigin GPU"
        return result

    def run_cpu(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            initial_temp=150.0,
            device="cpu",
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Rastrigin CPU-parallel"
        return result

    def run_sequential(self) -> dict:
        sa = ParallelSA(
            objective=self.objective,
            dim=self.dim,
            n_trials=self.n_trials,
            bounds=self.bounds,
            steps=self.steps,
            initial_temp=150.0,
            device="cpu",
            seed=42,
        )
        t0 = time.perf_counter()
        result = sa.run_sequential()
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        result["label"] = "Rastrigin CPU-sequential"
        return result
