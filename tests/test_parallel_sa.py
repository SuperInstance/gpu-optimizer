"""Tests for ParallelSA core engine."""

import pytest
import torch
import numpy as np
from gpu_optimizer.parallel_sa import ParallelSA


class TestParallelSA:
    """Tests for the core parallel SA engine."""

    def test_sphere_finds_near_optimum(self, sphere_objective):
        """On the sphere function, best_cost should be near 0."""
        sa = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=200,
            steps=1000,
            bounds=(-5.12, 5.12),
            seed=42,
            device="cpu",
        )
        result = sa.run()
        assert result["best_cost"] < 1.0, f"Expected cost < 1.0, got {result['best_cost']}"

    def test_returns_correct_shapes(self, sphere_objective):
        """Output arrays should have expected shapes."""
        sa = ParallelSA(
            objective=sphere_objective,
            dim=10,
            n_trials=50,
            steps=500,
            device="cpu",
            seed=0,
        )
        result = sa.run()
        assert result["best_solution"].shape == (10,)
        assert result["all_best_costs"].shape == (50,)
        assert result["history"].shape == (500,)

    def test_device_reported(self, sphere_objective):
        """Result should report the device used."""
        sa = ParallelSA(
            objective=sphere_objective,
            dim=3,
            n_trials=10,
            steps=100,
            device="cpu",
        )
        result = sa.run()
        assert result["device"] == "cpu"

    def test_sequential_matches_parallel_solution_quality(self, sphere_objective):
        """Sequential and parallel should both find reasonable solutions."""
        sa = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=50,
            steps=500,
            device="cpu",
            seed=42,
        )
        par = sa.run()
        seq = sa.run_sequential()
        # Both should get within reasonable range
        assert par["best_cost"] < 5.0
        assert seq["best_cost"] < 5.0

    def test_history_is_monotonically_non_increasing(self, sphere_objective):
        """Global best cost should never increase."""
        sa = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=50,
            steps=200,
            device="cpu",
            seed=7,
        )
        result = sa.run()
        hist = result["history"]
        for i in range(1, len(hist)):
            assert hist[i] <= hist[i - 1] + 1e-6, f"History increased at step {i}"

    def test_gpu_run_if_available(self, sphere_objective, gpu_available):
        """If GPU is available, run should use CUDA."""
        if not gpu_available:
            pytest.skip("No GPU available")
        sa = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=100,
            steps=200,
            seed=0,
        )
        result = sa.run()
        assert result["device"] == "cuda"
        assert result["best_cost"] < 2.0

    def test_bounds_respected(self):
        """Solutions should stay within bounds."""
        def obj(x):
            return (x ** 2).sum(dim=1)
        sa = ParallelSA(
            objective=obj,
            dim=5,
            n_trials=50,
            steps=200,
            bounds=(-1.0, 1.0),
            device="cpu",
            seed=0,
        )
        result = sa.run()
        assert result["best_solution"].min() >= -1.0 - 1e-6
        assert result["best_solution"].max() <= 1.0 + 1e-6

    def test_seed_reproducibility(self, sphere_objective):
        """Same seed should produce identical results."""
        sa1 = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=30,
            steps=100,
            device="cpu",
            seed=123,
        )
        sa2 = ParallelSA(
            objective=sphere_objective,
            dim=5,
            n_trials=30,
            steps=100,
            device="cpu",
            seed=123,
        )
        r1 = sa1.run()
        r2 = sa2.run()
        np.testing.assert_array_equal(r1["best_solution"], r2["best_solution"])
        assert r1["best_cost"] == r2["best_cost"]
