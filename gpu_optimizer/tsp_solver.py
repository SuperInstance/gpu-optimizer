"""TSP Solver using parallel simulated annealing on GPU."""

from __future__ import annotations

import torch
import numpy as np
import time
from typing import Optional


class TSPSolver:
    """Solve small TSP instances (10-30 cities) via parallel SA on GPU.

    Each trial maintains a permutation of cities.  Moves are 2-opt swaps
    (reversing a segment).  The entire batch of trials runs as tensor ops.

    Parameters
    ----------
    cities : np.ndarray
        ``(N_cities, 2)`` array of city coordinates.
    n_trials : int
        Number of parallel SA trials.
    initial_temp : float
        Starting temperature.
    final_temp : float
        Minimum temperature.
    cooling_rate : float
        Multiplicative cooling per step.
    steps : int
        Number of annealing iterations.
    device : str or None
        ``"cuda"`` if available, else ``"cpu"``.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        cities: np.ndarray,
        n_trials: int = 1000,
        initial_temp: float = 50.0,
        final_temp: float = 1e-3,
        cooling_rate: float = 0.998,
        steps: int = 3000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.cities = np.asarray(cities, dtype=np.float32)
        assert self.cities.ndim == 2 and self.cities.shape[1] == 2
        self.n_cities = self.cities.shape[0]
        assert 3 <= self.n_cities <= 200, "TSP solver designed for small instances"
        self.n_trials = n_trials
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.steps = steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    def _tour_distance_batch(self, tours: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute total tour distance for a batch of tours.

        Parameters
        ----------
        tours : LongTensor, shape ``(B, N)`` — permutation of city indices per trial.
        coords : FloatTensor, shape ``(N, 2)`` — city coordinates.

        Returns
        -------
        FloatTensor, shape ``(B,)``
        """
        # Gather coords: (B, N, 2)
        ordered = coords[tours]
        # Distance between consecutive cities
        diffs = ordered[:, 1:, :] - ordered[:, :-1, :]
        segment_dists = (diffs ** 2).sum(dim=2).sqrt()
        # Close the loop: last → first
        loop_diff = ordered[:, 0, :] - ordered[:, -1, :]
        loop_dist = (loop_diff ** 2).sum(dim=1).sqrt()
        return segment_dists.sum(dim=1) + loop_dist

    def run(self) -> dict:
        """Execute parallel SA for TSP.

        Returns
        -------
        dict with keys:
            best_tour : np.ndarray
            best_distance : float
            all_distances : np.ndarray
            device : str
            time : float
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dev = torch.device(self.device)
        coords = torch.tensor(self.cities, device=dev, dtype=torch.float32)
        n = self.n_cities
        B = self.n_trials

        # Initialise random tours: (B, N)
        tours = torch.stack([torch.randperm(n, device=dev) for _ in range(B)])
        costs = self._tour_distance_batch(tours, coords)

        best_tours = tours.clone()
        best_costs = costs.clone()

        temp = self.initial_temp

        t0 = time.perf_counter()

        for step in range(self.steps):
            # 2-opt move: pick two positions i < j, reverse segment [i:j+1]
            # Generate random pairs for all trials
            i_pos = torch.randint(0, n, (B,), device=dev)
            j_pos = torch.randint(0, n, (B,), device=dev)

            # Ensure i < j
            mask = i_pos > j_pos
            i_pos_new = torch.where(mask, j_pos, i_pos)
            j_pos_new = torch.where(mask, i_pos, j_pos)
            i_pos, j_pos = i_pos_new, j_pos_new

            # Skip trivial swaps (i == j or adjacent)
            valid = (j_pos - i_pos) > 1
            if not valid.any():
                continue

            # Build proposed tours (2-opt reversal)
            proposed = tours.clone()
            for b in range(B):
                if valid[b]:
                    ii, jj = i_pos[b].item(), j_pos[b].item()
                    proposed[b, ii:jj + 1] = tours[b, ii:jj + 1].flip(0)

            proposed_costs = self._tour_distance_batch(proposed, coords)

            # Metropolis acceptance
            delta = proposed_costs - costs
            accept_prob = torch.exp(-delta / temp).clamp(max=1.0)
            rand = torch.rand(B, device=dev)
            accept = (rand < accept_prob) & valid

            tours = torch.where(accept.unsqueeze(1), proposed, tours)
            costs = torch.where(accept, proposed_costs, costs)

            # Track best
            improved = costs < best_costs
            best_tours = torch.where(improved.unsqueeze(1), tours, best_tours)
            best_costs = torch.where(improved, costs, best_costs)

            temp = max(temp * self.cooling_rate, self.final_temp)

        elapsed = time.perf_counter() - t0

        global_idx = best_costs.argmin()
        best_tour = best_tours[global_idx].cpu().numpy()
        best_dist = best_costs[global_idx].cpu().item()

        return {
            "best_tour": best_tour,
            "best_distance": best_dist,
            "all_distances": best_costs.cpu().numpy(),
            "device": self.device,
            "time": elapsed,
        }

    def run_sequential(self) -> dict:
        """Run N trials sequentially on CPU for timing comparison."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dev = torch.device("cpu")
        coords = torch.tensor(self.cities, device=dev, dtype=torch.float32)
        n = self.n_cities
        all_dists = np.full(self.n_trials, np.inf)
        best_overall = np.inf
        best_tour = None

        t0 = time.perf_counter()
        for t in range(self.n_trials):
            tour = torch.randperm(n, device=dev).unsqueeze(0)
            cost = self._tour_distance_batch(tour, coords).item()
            trial_best = tour.clone()
            trial_best_cost = cost
            temp = self.initial_temp

            for _ in range(self.steps):
                i = torch.randint(0, n, (1,), device=dev).item()
                j = torch.randint(0, n, (1,), device=dev).item()
                if i > j:
                    i, j = j, i
                if j - i <= 1:
                    continue

                proposed = tour.clone()
                proposed[0, i:j + 1] = tour[0, i:j + 1].flip(0)
                new_cost = self._tour_distance_batch(proposed, coords).item()
                delta = new_cost - cost
                if np.random.random() < min(1.0, np.exp(-delta / temp)):
                    tour = proposed
                    cost = new_cost
                    if cost < trial_best_cost:
                        trial_best = tour.clone()
                        trial_best_cost = cost
                temp = max(temp * self.cooling_rate, self.final_temp)

            all_dists[t] = trial_best_cost
            if trial_best_cost < best_overall:
                best_overall = trial_best_cost
                best_tour = trial_best.squeeze().numpy()

        elapsed = time.perf_counter() - t0
        return {
            "best_tour": best_tour,
            "best_distance": best_overall,
            "all_distances": all_dists,
            "device": "cpu",
            "time": elapsed,
        }
