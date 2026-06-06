"""Shared test fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def gpu_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def sphere_objective():
    """Sphere function objective."""
    def obj(x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).sum(dim=1)
    return obj


@pytest.fixture
def random_cities():
    """Random 15-city TSP instance."""
    np.random.seed(42)
    return np.random.rand(15, 2).astype(np.float32) * 100


@pytest.fixture
def portfolio_data():
    """Sample portfolio data (5 assets)."""
    np.random.seed(42)
    returns = np.array([0.12, 0.10, 0.07, 0.03, 0.05], dtype=np.float32)
    cov = np.array([
        [0.040, 0.006, 0.002, -0.001, 0.003],
        [0.006, 0.025, 0.004, -0.002, 0.002],
        [0.002, 0.004, 0.016, 0.001, 0.001],
        [-0.001, -0.002, 0.001, 0.009, 0.000],
        [0.003, 0.002, 0.001, 0.000, 0.010],
    ], dtype=np.float32)
    return returns, cov
