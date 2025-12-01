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
            "x": 0.5 * np.array([[0,1],[1, 0]], dtype="complex128"),
            "y": 0.5 * np.array([[0,-1j], [1j,0]], dtype="complex128"),
            "z": 0.5 * np.array([[1, 0], [0, -1]], dtype="complex128")
        }

        self.H = self.compute_H()
        self.exp_H = expm(-self.problem.beta * self.H)

    def compute_S(self, i, dir):
        """Returns $S^dir_i S^dir_{i+1}$."""
        S = self.S[dir] if i == 0 else np.eye(2)
        for j in range(1, self.problem.n_sites):
            this_S = self.S[dir] if i == j or i == j+1 else np.eye(2)
            S = np.kron(S, this_S)
        return S

    def compute_H(self):
        dim_H = 1 << self.problem.n_sites
        H = np.zeros((dim_H, dim_H), dtype="complex128")
        for i in range(self.problem.n_sites):
            H += self.problem.J_x * self.compute_S(i, "x")
            H += self.problem.J_x * self.compute_S(i, "y")
            H += self.problem.J_z * self.compute_S(i, "z")
        return H
    
    def compute(self, obs):
        """
        Computes the expected value of the observable provided.
        $<O>=tr(e^{-\\beta H}O)/tr(e^{\\beta H})$
        """
        return np.trace(self.exp_H * obs) / np.trace(self.exp_H)
    
    @functools.cached_property
    def energy(self):
        return self.compute(self.H)
    
    def all_configs(self):
        # initial_config = np.zeros((2*self.problem.m, self.problem.n_sites))

        # def aux(i, j, config):
        #     if j == 2*self.problem.m:
        #         return [config.copy()]
        #     if i == self.problem.n_sites:
        #         return aux(0, j+1, config)
        #     config[j, i] = 1
        #     res = aux(i+1, j, config)
        #     config[j, i] = -1
        #     return res + aux(i+1, j, config)

        # return aux(0, 0, initial_config)
        total_configs = 2 * self.problem.m * self.problem.n_sites
        flats_configs = itertools.product([1, -1], repeat=total_configs)

        for flat_config in tqdm(flats_configs, total=total_configs, desc="Generating configs"):
            config = np.array(flat_config).reshape((2*self.problem.m), self.problem.n_sites)
            
            yield config

