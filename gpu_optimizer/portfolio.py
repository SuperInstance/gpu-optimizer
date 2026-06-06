"""Portfolio optimizer: minimize risk (variance) subject to return constraint."""

from __future__ import annotations

import torch
import numpy as np
import time
from typing import Optional


class PortfolioOptimizer:
    """Minimize portfolio variance subject to a minimum expected return.

    The optimisation variable is the weight vector **w** ∈ ℝᴺ (one weight per
    asset).  We encode the problem as an unconstrained SA over softmax-
    normalised weights and penalise violations of the return constraint.

    Parameters
    ----------
    returns : np.ndarray
        ``(N_assets,)`` expected (mean) returns for each asset.
    cov_matrix : np.ndarray
        ``(N_assets, N_assets)`` covariance matrix of asset returns.
    min_return : float
        Minimum acceptable expected portfolio return.
    n_trials : int
        Number of parallel SA trials.
    steps : int
        Number of annealing iterations.
    penalty : float
        Penalty multiplier for violating the return constraint.
    device : str or None
        ``"cuda"`` if available, else ``"cpu"``.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_return: float = 0.0,
        n_trials: int = 1000,
        steps: int = 2000,
        penalty: float = 100.0,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.returns = np.asarray(returns, dtype=np.float32)
        self.cov = np.asarray(cov_matrix, dtype=np.float32)
        self.n_assets = len(returns)
        assert self.cov.shape == (self.n_assets, self.n_assets)
        self.min_return = min_return
        self.n_trials = n_trials
        self.steps = steps
        self.penalty = penalty
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    def _objective(self, logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute penalised portfolio cost for a batch.

        Parameters
        ----------
        logits : Tensor ``(B, N)`` – unnormalised log-weights.
        mu : Tensor ``(N,)`` – expected returns.
        sigma : Tensor ``(N, N)`` – covariance matrix.

        Returns
        -------
        Tensor ``(B,)`` – penalised cost per trial.
        """
        # Softmax → valid weight vectors
        w = torch.softmax(logits, dim=1)  # (B, N)
        # Portfolio variance = w^T Σ w
        port_var = (w @ sigma * w).sum(dim=1)  # (B,)
        # Expected return = w^T μ
        port_ret = w @ mu  # (B,)
        # Penalty for return shortfall
        shortfall = torch.relu(self.min_return - port_ret)
        return port_var + self.penalty * shortfall

    def run(self) -> dict:
        """Execute parallel SA for portfolio optimisation.

        Returns
        -------
        dict with keys:
            best_weights : np.ndarray  – optimal weight allocation
            best_cost     : float
            portfolio_variance : float
            portfolio_return : float
            all_costs : np.ndarray
            device : str
            time : float
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dev = torch.device(self.device)
        mu = torch.tensor(self.returns, device=dev)
        sigma = torch.tensor(self.cov, device=dev)
        B = self.n_trials
        N = self.n_assets

        # Start with random logits (softmax gives random valid weights)
        current = torch.randn(B, N, device=dev)
        current_cost = self._objective(current, mu, sigma)

        best = current.clone()
        best_cost = current_cost.clone()

        temp = 50.0
        cooling_rate = 0.997
        final_temp = 1e-4

        t0 = time.perf_counter()
        for step in range(self.steps):
            scale = (temp / 50.0) * 0.5
            noise = torch.randn_like(current) * scale
            proposal = current + noise
            proposal_cost = self._objective(proposal, mu, sigma)

            delta = proposal_cost - current_cost
            accept_prob = torch.exp(-delta / temp).clamp(max=1.0)
            accept = torch.rand(B, device=dev) < accept_prob

            current = torch.where(accept.unsqueeze(1), proposal, current)
            current_cost = torch.where(accept, proposal_cost, current_cost)

            improved = current_cost < best_cost
            best = torch.where(improved.unsqueeze(1), current, best)
            best_cost = torch.where(improved, current_cost, best_cost)

            temp = max(temp * cooling_rate, final_temp)

        elapsed = time.perf_counter() - t0

        global_idx = best_cost.argmin()
        best_logits = best[global_idx]
        w = torch.softmax(best_logits, dim=0).cpu().numpy()

        port_var = float(w @ self.cov @ w)
        port_ret = float(w @ self.returns)

        return {
            "best_weights": w,
            "best_cost": best_cost[global_idx].cpu().item(),
            "portfolio_variance": port_var,
            "portfolio_return": port_ret,
            "all_costs": best_cost.cpu().numpy(),
            "device": self.device,
            "time": elapsed,
        }

    def run_sequential(self) -> dict:
        """Run N trials sequentially on CPU."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dev = torch.device("cpu")
        mu = torch.tensor(self.returns, device=dev)
        sigma = torch.tensor(self.cov, device=dev)
        N = self.n_assets

        all_costs = np.full(self.n_trials, np.inf)
        best_overall_cost = np.inf
        best_overall_logits = None

        t0 = time.perf_counter()
        for t in range(self.n_trials):
            current = torch.randn(1, N, device=dev)
            current_cost = self._objective(current, mu, sigma)
            trial_best = current.clone()
            trial_best_cost = current_cost.clone()
            temp = 50.0

            for _ in range(self.steps):
                scale = (temp / 50.0) * 0.5
                proposal = current + torch.randn(1, N, device=dev) * scale
                proposal_cost = self._objective(proposal, mu, sigma)
                delta = proposal_cost - current_cost
                if torch.rand(1, device=dev) < torch.exp(-delta / temp).clamp(max=1.0):
                    current = proposal
                    current_cost = proposal_cost
                    if current_cost < trial_best_cost:
                        trial_best = current.clone()
                        trial_best_cost = current_cost.clone()
                temp = max(temp * 0.997, 1e-4)

            cost_val = trial_best_cost.item()
            all_costs[t] = cost_val
            if cost_val < best_overall_cost:
                best_overall_cost = cost_val
                best_overall_logits = trial_best.squeeze()

        elapsed = time.perf_counter() - t0

        w = torch.softmax(best_overall_logits, dim=0).numpy()
        port_var = float(w @ self.cov @ w)
        port_ret = float(w @ self.returns)

        return {
            "best_weights": w,
            "best_cost": best_overall_cost,
            "portfolio_variance": port_var,
            "portfolio_return": port_ret,
            "all_costs": all_costs,
            "device": "cpu",
            "time": elapsed,
        }
