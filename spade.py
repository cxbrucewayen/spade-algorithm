"""
spade.py — SPADE: Sample-based Pre-Assessment Differential Evolution

Direct Python translation of SPADE.m (MATLAB prototype) with full type
annotations and the unified BaseAlgorithm interface.

Five Improvement Mechanisms (M1–M5)
------------------------------------
M1: Three-track heterogeneous mutation strategy
      Track-1: current-to-pbest/1 + binomial crossover  (CR1=0.9, F1~U[0.5,1.0])
      Track-2: rand/1 + binomial crossover               (CR2=0.2, F2~U[0.8,1.0])
      Track-3: current-to-rand/1 (no crossover)          (F3~U[0.5,1.0])
              克服二项式交叉的坐标偏见，有效应对旋转非分离函数
M2: kNN surrogate pre-screening (m reference samples, zero training cost)
      轻量级近邻代理预筛选，1-NN 最近邻评分，O(m·D) 复杂度
M3: Composite surrogate score: Score = f_norm - alpha * d_norm
      最近邻适应度估计与多样性距离奖励融合，利用 m 个参考样本中最近者的信息
M4: Midpoint bounce boundary handling
      u[j] = (x[j] + bound[j]) / 2，将越界能量转化为可行域内探索


References
----------
SPADE.m — MATLAB prototype (SPADE/SPADE.m in this repository)
SPADE_strategy.txt — Design rationale (SPADE/SPADE_strategy.txt)
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .base import BaseAlgorithm, OptimizationResult


class SPADE(BaseAlgorithm):
    """Sample-based Pre-Assessment Differential Evolution.

    Inherits unified interface from BaseAlgorithm:
      - FEs counted exclusively via self._evaluate()
      - Convergence curve recorded via self._record_convergence()
      - Population initialised via self._init_population()
      - Results packaged as OptimizationResult

    Parameters
    ----------
    dim : int
        Search space dimensionality.
    lb : array-like of shape (dim,)
        Lower bounds.
    ub : array-like of shape (dim,)
        Upper bounds.
    pop_size : int, default 30
        Population size N.
    max_fes : int, default 300_000
        Maximum function evaluations (hard termination criterion).
    seed : int, default 42
        Random seed for full reproducibility across runs.
    m : int, default 5
        Number of reference samples used in kNN surrogate (M2).
        Automatically clamped to [1, pop_size].
    alpha : float, default 0.1
        Diversity weight in composite surrogate score
        S = f_norm - alpha * d_norm  (M3).
    top_p : float, default 0.1
        Fraction of population forming the elite pool for pbest (M1, Track-1).
        n_elite = max(1, round(top_p * N)).
    F1_range : tuple[float, float], default (0.5, 1.0)
        Uniform sampling range for mutation factor F1 (Track-1).
    CR1 : float, default 0.9
        Crossover rate for Track-1 (current-to-pbest/1 + binomial).
    F2_range : tuple[float, float], default (0.8, 1.0)
        Uniform sampling range for mutation factor F2 (Track-2).
    CR2 : float, default 0.2
        Crossover rate for Track-2 (rand/1 + binomial).
    F3_range : tuple[float, float], default (0.5, 1.0)
        Uniform sampling range for mutation factor F3 (Track-3).
        Track-3 uses no crossover operator (rotation-invariant by design).
    """

    def __init__(
        self,
        dim: int,
        lb: np.ndarray | list[float],
        ub: np.ndarray | list[float],
        pop_size: int = 30,
        max_fes: int = 300_000,
        seed: int = 42,
        # --- SPADE-specific hyperparameters ---
        m: int = 5,
        alpha: float = 0.1,
        top_p: float = 0.1,
        F1_range: tuple[float, float] = (0.5, 1.0),
        CR1: float = 0.9,
        F2_range: tuple[float, float] = (0.8, 1.0),
        CR2: float = 0.2,
        F3_range: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        super().__init__(dim, lb, ub, pop_size, max_fes, seed)

        # Clamp m to valid range
        self.m: int = max(1, min(m, pop_size))
        self.alpha: float = alpha
        self.top_p: float = top_p
        self.F1_range: tuple[float, float] = F1_range
        self.CR1: float = CR1
        self.F2_range: tuple[float, float] = F2_range
        self.CR2: float = CR2
        self.F3_range: tuple[float, float] = F3_range

    # ------------------------------------------------------------------
    # Public interface — required by BaseAlgorithm
    # ------------------------------------------------------------------

    def optimize(self, func: Callable[[np.ndarray], float]) -> OptimizationResult:
        """Run SPADE optimisation.

        Faithfully implements the five-mechanism logic from SPADE.m translated
        to NumPy vectorised idioms. All function evaluations go through
        self._evaluate() to ensure accurate FEs accounting and global-best
        tracking.

        Args:
            func: Objective function f(x) -> float. Each call counts as 1 FE.

        Returns:
            OptimizationResult with full convergence history and timings.
        """
        self._reset_state()
        t_start = time.perf_counter()

        N = self.pop_size
        D = self.dim

        # ------------------------------------------------------------------
        # Step 0: Population initialisation
        # ------------------------------------------------------------------
        pop = self._init_population()          # shape (N, D)
        fitness = np.empty(N, dtype=float)
        for i in range(N):
            fitness[i] = self._evaluate(func, pop[i])

        # Record convergence after initialisation
        self._record_convergence()

        # Diagonal length of the search space — normalisation constant for M3
        # Diag = ||ub - lb||_2  (computed once, constant throughout)
        diag: float = float(np.linalg.norm(self.ub - self.lb))

        # ------------------------------------------------------------------
        # Main evolutionary loop (FEs-controlled, mirrors SPADE.m while-loop)
        # ------------------------------------------------------------------
        while self._fes < self.max_fes:
            # Population-level statistics for surrogate normalisation (M3)
            min_cost = float(fitness.min())
            max_cost = float(fitness.max())

            # M1: Elite pool for pbest selection (Track-1)
            n_elite = max(1, round(self.top_p * N))
            elite_idx = np.argsort(fitness)[:n_elite]   # indices of n_elite best

            for i in range(N):
                if self._fes >= self.max_fes:
                    break

                x = pop[i]  # current individual (parent)

                # --- Select r1, r2, r3: distinct indices, all ≠ i ---
                pool = np.delete(np.arange(N), i)
                r1, r2, r3 = self.rng.choice(pool, size=3, replace=False)

                # pbest: uniform random draw from elite pool
                pbest = pop[self.rng.choice(elite_idx)]

                # ============================================================
                # M1: Three-track heterogeneous mutation (Track-1/2/3)
                # ============================================================

                # Track-1: current-to-pbest/1 + binomial crossover
                # v1 = x + F1*(pbest - x) + F1*(x_r1 - x_r2)
                F1 = self.rng.uniform(self.F1_range[0], self.F1_range[1])
                v1 = x + F1 * (pbest - x) + F1 * (pop[r1] - pop[r2])
                u1 = self._binomial_crossover(x, v1, self.CR1)

                # Track-2: rand/1 + binomial crossover
                # v2 = x_r1 + F2*(x_r2 - x_r3)
                F2 = self.rng.uniform(self.F2_range[0], self.F2_range[1])
                v2 = pop[r1] + F2 * (pop[r2] - pop[r3])
                u2 = self._binomial_crossover(x, v2, self.CR2)

                # Track-3: current-to-rand/1 (no crossover, rotation-invariant)
                # u3 = x + rw*(x_r1 - x) + F3*(x_r2 - x_r3)
                F3 = self.rng.uniform(self.F3_range[0], self.F3_range[1])
                rw = self.rng.uniform()
                u3 = x + rw * (pop[r1] - x) + F3 * (pop[r2] - pop[r3])

                # ============================================================
                # M4: Midpoint bounce boundary handling (all 3 candidates)
                # 越界时取父代坐标与边界的中点，保留探索能量
                # ============================================================
                u1 = self._midpoint_bounce(u1, x)
                u2 = self._midpoint_bounce(u2, x)
                u3 = self._midpoint_bounce(u3, x)

                # ============================================================
                # M2: kNN surrogate pre-screening
                # Randomly sample m reference individuals (O(m·D) cost)
                # ============================================================
                ref_idx = self.rng.choice(N, size=self.m, replace=False)
                ref_pos = pop[ref_idx]       # shape (m, D)
                ref_fit = fitness[ref_idx]   # shape (m,)

                # M3: Compute composite surrogate score for each candidate
                scores = np.array([
                    self._surrogate_score(
                        u, ref_pos, ref_fit, min_cost, max_cost, diag
                    )
                    for u in (u1, u2, u3)
                ])

                # Select candidate with minimum score (best surrogate quality)
                best_k = int(np.argmin(scores))
                u_selected: np.ndarray = (u1, u2, u3)[best_k]

                # ============================================================

                # 严格单次真实评估，禁止代理Score直接进入进化更新
                # ============================================================
                new_fitness = self._evaluate(func, u_selected)

                # Greedy one-on-one selection (mirrors SPADE.m update logic)
                if new_fitness < fitness[i]:
                    pop[i] = u_selected.copy()
                    fitness[i] = new_fitness

            # Record convergence once per generation
            self._record_convergence()

        # ------------------------------------------------------------------
        # Build and return standardised result
        # ------------------------------------------------------------------
        wall_time = time.perf_counter() - t_start
        return OptimizationResult(
            best_fitness=self._best_fitness,
            best_position=self._best_position.copy(),
            convergence_curve=list(self._convergence),
            convergence_fes=list(self._convergence_fes),
            total_fes=self._fes,
            wall_time=wall_time,
        )

    def get_params(self) -> dict:
        """Return full hyperparameter dict including SPADE-specific parameters.

        Used by result loggers to annotate output files.

        Returns:
            Dict with base params (name, dim, pop_size, max_fes, seed) plus
            SPADE-specific: m, alpha, top_p, F1_range, CR1, F2_range, CR2,
            F3_range.
        """
        params = super().get_params()
        params.update(
            {
                "m": self.m,
                "alpha": self.alpha,
                "top_p": self.top_p,
                "F1_range": self.F1_range,
                "CR1": self.CR1,
                "F2_range": self.F2_range,
                "CR2": self.CR2,
                "F3_range": self.F3_range,
            }
        )
        return params

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _binomial_crossover(
        self, x: np.ndarray, v: np.ndarray, CR: float
    ) -> np.ndarray:
        """Binomial (uniform) crossover between parent x and donor vector v.

        Guarantees at least one dimension is inherited from v via a mandatory
        change index j0 (mirrors SPADE.m lines: j0 = randi(dim); ...).

        Args:
            x:  Parent individual, shape (D,).
            v:  Donor (mutant) vector, shape (D,).
            CR: Crossover probability in (0, 1].

        Returns:
            Trial vector u of shape (D,).
        """
        u = x.copy()
        j0 = self.rng.randint(0, self.dim)          # mandatory-change index
        mask = self.rng.uniform(size=self.dim) <= CR
        mask[j0] = True                              # force at least one gene from v
        u[mask] = v[mask]
        return u

    def _midpoint_bounce(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply midpoint bounce boundary correction (M4).

        For each dimension j:
          - u[j] < lb[j]  →  u[j] = (x[j] + lb[j]) / 2
          - u[j] > ub[j]  →  u[j] = (x[j] + ub[j]) / 2

        This replaces hard-clamp truncation with a midpoint inside the feasible
        region, preserving exploration potential.  x is always feasible, so
        (x[j] + bound[j]) / 2 is guaranteed to lie within [lb[j], ub[j]].

        Args:
            u: Trial vector (possibly out-of-bounds), shape (D,).
            x: Parent individual (always feasible), shape (D,).

        Returns:
            Feasibility-corrected trial vector of shape (D,).
        """
        u = u.copy()
        below = u < self.lb
        above = u > self.ub
        if below.any():
            u[below] = (x[below] + self.lb[below]) * 0.5
        if above.any():
            u[above] = (x[above] + self.ub[above]) * 0.5
        return u

    def _surrogate_score(
        self,
        candidate: np.ndarray,
        ref_positions: np.ndarray,
        ref_fitness: np.ndarray,
        min_cost: float,
        max_cost: float,
        diag: float,
    ) -> float:
        """Compute 1-NN surrogate score for a candidate (M2+M3).

        Finds the nearest reference individual (1-NN) among the m randomly
        sampled references, then computes a composite score combining the
        nearest neighbour's fitness and the distance to it.

        This mirrors the original MATLAB implementation (SPADE.m lines 76-84):
          nearestCost = pop(refIdx(argmin)).Cost
          minDist     = min distance to any reference
          Score = (nearestCost - minCost) / (maxCost - minCost + eps)
                  - alpha * (minDist / (Diag + eps))

        Lower score → better surrogate quality (lower cost + higher diversity).

        Args:
            candidate:     Candidate vector to score, shape (D,).
            ref_positions: Reference individual positions, shape (m, D).
            ref_fitness:   Reference individual fitness values, shape (m,).
            min_cost:      Population minimum fitness (normalisation).
            max_cost:      Population maximum fitness (normalisation).
            diag:          Search-space diagonal length (normalisation constant).

        Returns:
            Scalar surrogate score (lower is better).
        """
        # Euclidean distances from candidate to all m reference individuals
        dists = np.linalg.norm(ref_positions - candidate, axis=1)  # shape (m,)

        # 1-NN: find nearest reference
        nn_idx = int(np.argmin(dists))
        nearest_cost = ref_fitness[nn_idx]
        min_dist = dists[nn_idx]

        # Composite score: normalised nearest fitness - alpha * normalised distance
        f_norm = (nearest_cost - min_cost) / (max_cost - min_cost + 1e-15)
        d_norm = min_dist / (diag + 1e-15)
        return f_norm - self.alpha * d_norm
