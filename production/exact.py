import numpy as np
from scipy.linalg import expm
from production.problem import Problem
import itertools
from tqdm import tqdm
import functools


class ExactSolver:
    """
    Solves the problem using the $2^n$ matrices.

    Arguments:
    - problem (Problem): The settings for the solver.
    """

    def __init__(self, problem: Problem):
        self.problem = problem

        # Base matricies $S^x, S^y, S^z$
        self.S = {
            "x": 0.5 * np.array([[0, 1], [1, 0]], dtype="complex128"),
            "y": 0.5 * np.array([[0, -1j], [1j, 0]], dtype="complex128"),
            "z": 0.5 * np.array([[1, 0], [0, -1]], dtype="complex128"),
        }

        self.H = self.compute_H()
        self.exp_H = expm(-self.problem.beta * self.H)

    def compute_S(self, i, ax):
        """Returns $S^dir_i S^dir_{i+1}$."""
        # S = self.S[dir] if i == 0 else np.eye(2)
        # for j in range(1, self.problem.n_sites):
        #     this_S = self.S[dir] if i == j or i == j+1 else np.eye(2)
        #     S = np.kron(S, this_S)
        n = self.problem.n_sites
        ops = [np.eye(2, dtype="complex128") for _ in range(n)]
        ops[i] = self.S[ax]
        ops[(i + 1) % n] = self.S[ax]

        S = ops[0]
        for op in ops[1:]:
            S = np.kron(S, op)

        return S

    def compute_H(self):
        dim_H = 1 << self.problem.n_sites
        H = np.zeros((dim_H, dim_H), dtype="complex128")
        for i in range(self.problem.n_sites):
            H += self.problem.J_x * self.compute_S(i, "x")
            H += self.problem.J_x * self.compute_S(i, "y")
            H += self.problem.J_z * self.compute_S(i, "z")
        return H

    def get_sector_indices(self, n_up):
        """
        Identifies the indices in the 2^N Hilbert space that correspond
        to states with exactly n_up spins.
        """
        indices = []
        # Iterate through all 2^N basis states
        for i in range(1 << self.problem.n_sites):
            # Check population count (number of set bits)
            # In binary representation, 1=Up, 0=Down
            if bin(i).count("1") == n_up:
                indices.append(i)
        return np.array(indices)

    def compute(self, obs, n_up=None):
        """
        Computes <O> = tr(rho * O) / tr(rho).

        If n_up is provided, restricts the trace to the subspace with
        fixed number of up spins.
        """
        if n_up is None:
            # Full Hilbert Space Average (Grand Canonical / Open Shell)
            return np.trace(self.exp_H @ obs) / np.trace(self.exp_H)
        else:
            # Canonical Ensemble Average (Fixed Magnetization)
            idx = self.get_sector_indices(n_up)

            if len(idx) == 0:
                raise ValueError(f"No states found with n_up={n_up}")

            # Project matrices onto the subspace defined by idx
            # np.ix_ creates a mesh to extract the submatrix where row,col are in idx
            sub_rho = self.exp_H[np.ix_(idx, idx)]
            sub_obs = obs[np.ix_(idx, idx)]

            # Compute trace within this block
            val = np.trace(sub_rho @ sub_obs) / np.trace(sub_rho)
            return val

    @functools.cached_property
    def energy(self):
        return self.compute(self.H)

    def all_configs(self):
        total_configs = 2 * self.problem.m * self.problem.n_sites
        flats_configs = itertools.product([1, -1], repeat=total_configs)

        for flat_config in tqdm(
            flats_configs, total=total_configs, desc="Generating configs"
        ):
            config = np.array(flat_config).reshape(
                (2 * self.problem.m), self.problem.n_sites
            )

            yield config
