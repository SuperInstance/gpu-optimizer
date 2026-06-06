"""Tests for portfolio optimizer."""

import pytest
import numpy as np
from gpu_optimizer.portfolio import PortfolioOptimizer


class TestPortfolioOptimizer:
    def test_weights_sum_to_one(self, portfolio_data):
        returns, cov = portfolio_data
        opt = PortfolioOptimizer(returns, cov, n_trials=100, steps=500, device="cpu", seed=42)
        result = opt.run()
        assert abs(result["best_weights"].sum() - 1.0) < 1e-5

    def test_weights_non_negative(self, portfolio_data):
        returns, cov = portfolio_data
        opt = PortfolioOptimizer(returns, cov, n_trials=100, steps=500, device="cpu", seed=42)
        result = opt.run()
        assert (result["best_weights"] >= -1e-6).all()

    def test_return_constraint_satisfied(self, portfolio_data):
        returns, cov = portfolio_data
        min_ret = 0.06
        opt = PortfolioOptimizer(
            returns, cov, min_return=min_ret, n_trials=200, steps=1000,
            device="cpu", seed=42, penalty=200.0,
        )
        result = opt.run()
        assert result["portfolio_return"] >= min_ret - 0.01  # tolerance

    def test_variance_positive(self, portfolio_data):
        returns, cov = portfolio_data
        opt = PortfolioOptimizer(returns, cov, n_trials=50, steps=200, device="cpu", seed=0)
        result = opt.run()
        assert result["portfolio_variance"] > 0

    def test_sequential_runs(self, portfolio_data):
        returns, cov = portfolio_data
        opt = PortfolioOptimizer(returns, cov, n_trials=10, steps=100, device="cpu", seed=0)
        result = opt.run_sequential()
        assert abs(result["best_weights"].sum() - 1.0) < 1e-5
        assert result["time"] > 0
