"""Tests for TSP solver."""

import pytest
import numpy as np
from gpu_optimizer.tsp_solver import TSPSolver


class TestTSPSolver:
    def test_finds_valid_tour(self, random_cities):
        solver = TSPSolver(random_cities, n_trials=50, steps=500, device="cpu", seed=42)
        result = solver.run()
        tour = result["best_tour"]
        # Tour must be a permutation of all cities
        assert sorted(tour.tolist()) == list(range(len(random_cities)))
        assert result["best_distance"] > 0

    def test_sequential_finds_valid_tour(self, random_cities):
        solver = TSPSolver(random_cities, n_trials=10, steps=200, device="cpu", seed=42)
        result = solver.run_sequential()
        tour = result["best_tour"]
        assert sorted(tour.tolist()) == list(range(len(random_cities)))

    def test_distance_calculation(self):
        """Known triangle: 3 cities at (0,0), (1,0), (0,1). Best tour ≈ 2 + √2."""
        cities = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        solver = TSPSolver(cities, n_trials=100, steps=1000, device="cpu", seed=42)
        result = solver.run()
        # Optimal tour: 0→1→2→0 = 1 + √2 + 1 ≈ 3.414
        assert result["best_distance"] < 4.0

    def test_timing_reported(self, random_cities):
        solver = TSPSolver(random_cities, n_trials=20, steps=100, device="cpu", seed=0)
        result = solver.run()
        assert result["time"] > 0
