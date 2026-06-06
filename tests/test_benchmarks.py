"""Tests for benchmark functions."""

import pytest
import numpy as np
from gpu_optimizer.benchmarks import SphereBenchmark, RastriginBenchmark


class TestSphereBenchmark:
    def test_gpu_finds_low_cost(self):
        bench = SphereBenchmark(dim=5, n_trials=200, steps=1000)
        result = bench.run_gpu() if pytest.importorskip("torch").cuda.is_available() else bench.run_cpu()
        assert result["best_cost"] < 1.0

    def test_cpu_finds_low_cost(self):
        bench = SphereBenchmark(dim=5, n_trials=200, steps=1000)
        result = bench.run_cpu()
        assert result["best_cost"] < 1.0

    def test_timing_reported(self):
        bench = SphereBenchmark(dim=3, n_trials=50, steps=200)
        result = bench.run_cpu()
        assert "time" in result
        assert result["time"] > 0

    def test_sequential_runs(self):
        bench = SphereBenchmark(dim=3, n_trials=10, steps=100)
        result = bench.run_sequential()
        assert result["best_cost"] < 5.0
        assert result["time"] > 0


class TestRastriginBenchmark:
    def test_cpu_finds_reasonable_solution(self):
        bench = RastriginBenchmark(dim=5, n_trials=200, steps=1500)
        result = bench.run_cpu()
        # Rastrigin is hard; just verify it finds something < 50
        assert result["best_cost"] < 50.0

    def test_label_set(self):
        bench = RastriginBenchmark(dim=3, n_trials=10, steps=100)
        result = bench.run_cpu()
        assert "label" in result
        assert "Rastrigin" in result["label"]
