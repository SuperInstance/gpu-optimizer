"""Core parallel simulated annealing engine using PyTorch."""

from __future__ import annotations

import torch
import numpy as np
from typing import Callable, Optional


class ParallelSA:
    """Run N simulated annealing trials in parallel on GPU (or CPU fallback).

    Each trial independently explores the solution space, and the best across
    all trials is returned. The entire batch of trials is executed as tensor
    operations, enabling massive parallelism on CUDA.

    Parameters
    ----------
    objective : callable
        Function mapping ``(N, D)`` tensor of candidates → ``(N,)`` tensor of costs.
    dim : int
        Dimensionality of the search space.
    n_trials : int
        Number of parallel SA trials.
    bounds : tuple[float, float]
        Lower and upper bounds for each dimension.
    initial_temp : float
        Starting temperature.
    final_temp : float
        Minimum temperature (annealing stops here).
    cooling_rate : float
        Multiplicative cooling factor per step (0 < rate < 1).
    steps : int
        Number of annealing iterations.
    device : str or None
        ``"cuda"`` if available, else ``"cpu"``.  Auto-detected when *None*.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        n_trials: int = 1000,
        bounds: tuple[float, float] = (-5.12, 5.12),
        initial_temp: float = 100.0,
        final_temp: float = 1e-4,
        cooling_rate: float = 0.997,
        steps: int = 2000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.objective = objective
        self.dim = dim
        self.n_trials = n_trials
        self.bounds = bounds
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.steps = steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Execute parallel simulated annealing and return results.

        Returns
        -------
        dict with keys:
            best_solution : np.ndarray  – best candidate found (shape ``(D,)``)
            best_cost     : float        – cost of *best_solution*
            all_best_costs: np.ndarray   – per-trial best cost (shape ``(N,)``)
            history       : np.ndarray   – global best cost per step (shape ``(steps,)``)
            device        : str          – device used
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dev = torch.device(self.device)
        lo, hi = self.bounds

        # Initialise N random candidates
        current = torch.rand(self.n_trials, self.dim, device=dev) * (hi - lo) + lo
        current_cost = self.objective(current)

        best = current.clone()
        best_cost = current_cost.clone()

        temp = self.initial_temp
        history = torch.empty(self.steps, device=dev)

        for step in range(self.steps):
            # Propose perturbations — scaled to remaining temperature
            scale = (temp / self.initial_temp) * (hi - lo) * 0.1
            noise = torch.randn_like(current) * scale
            proposal = current + noise

            # Clamp to bounds
            proposal = proposal.clamp(lo, hi)
            proposal_cost = self.objective(proposal)

            # Metropolis acceptance
            delta = proposal_cost - current_cost
            accept_prob = torch.exp(-delta / temp)
            accept_prob = accept_prob.clamp(max=1.0)
            rand = torch.rand(self.n_trials, device=dev)
            accept_mask = rand < accept_prob

            # Update accepted proposals
            current = torch.where(
                accept_mask.unsqueeze(1), proposal, current
            )
            current_cost = torch.where(accept_mask, proposal_cost, current_cost)

            # Track per-trial best
            improved = proposal_cost < best_cost
            best = torch.where(improved.unsqueeze(1), proposal, best)
            best_cost = torch.where(improved, proposal_cost, best_cost)

            # Actually we need to also accept SA moves that might not improve
            # but current already has accepted moves. Re-evaluate best:
            improved2 = current_cost < best_cost
            best = torch.where(improved2.unsqueeze(1), current, best)
            best_cost = torch.where(improved2, current_cost, best_cost)

            history[step] = best_cost.min()
            temp = max(temp * self.cooling_rate, self.final_temp)

        # Global best
        global_best_idx = best_cost.argmin()
        result = {
            "best_solution": best[global_best_idx].cpu().numpy(),
            "best_cost": best_cost[global_best_idx].cpu().item(),
            "all_best_costs": best_cost.cpu().numpy(),
            "history": history.cpu().numpy(),
            "device": self.device,
        }
        return result

    def run_sequential(self) -> dict:
        """Run N trials sequentially (CPU-only) for comparison.

        Same algorithm as :meth:`run` but executes one trial at a time on CPU.
        """
        dev = torch.device("cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)

        lo, hi = self.bounds
        all_best = np.full(self.n_trials, np.inf)
        best_overall_cost = np.inf
        best_overall_sol = None

        for i in range(self.n_trials):
            current = torch.rand(1, self.dim, device=dev) * (hi - lo) + lo
            current_cost = self.objective(current)
            trial_best = current.clone()
            trial_best_cost = current_cost.clone()

            temp = self.initial_temp
            for _ in range(self.steps):
                scale = (temp / self.initial_temp) * (hi - lo) * 0.1
                noise = torch.randn(1, self.dim, device=dev) * scale
                proposal = (current + noise).clamp(lo, hi)
                proposal_cost = self.objective(proposal)
                delta = proposal_cost - current_cost
                if torch.rand(1, device=dev) < torch.exp(-delta / temp).clamp(max=1.0):
                    current = proposal
                    current_cost = proposal_cost
                if current_cost < trial_best_cost:
                    trial_best = current.clone()
                    trial_best_cost = current_cost.clone()
                temp = max(temp * self.cooling_rate, self.final_temp)

            cost_val = trial_best_cost.item()
            all_best[i] = cost_val
            if cost_val < best_overall_cost:
                best_overall_cost = cost_val
                best_overall_sol = trial_best.squeeze().numpy()

        return {
            "best_solution": best_overall_sol,
            "best_cost": best_overall_cost,
            "all_best_costs": all_best,
            "device": "cpu",
        }
