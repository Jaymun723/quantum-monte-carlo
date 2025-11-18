import numpy as np
from scipy.linalg import expm

class ExactSolver:
    def __init__(self, problem):
        self.problem = problem

        self.S = {
            "x": 0.5 * np.array([[0,1],[1, 0]], dtype="complex128"),
            "y": 0.5 * np.array([[0,-1j], [1j,0]], dtype="complex128"),
            "z": 0.5 * np.array([[1, 0], [0, -1]], dtype="complex128")
        }

        self.H = self.compute_H()
        self.exp_H = expm(-self.problem.beta * self.H)

    def compute_S(self, i, dir):
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
        return np.trace(self.exp_H * obs) / np.trace(self.exp_H)
    
    def energy(self):
        return self.compute(self.H)
