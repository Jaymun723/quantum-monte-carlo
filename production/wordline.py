from problem import Problem
import numpy as np

class Wordline: # "w"
    def __init__(self, problem: Problem):
        self.problem = problem

    def weight(self): # Omega(w)
        # `P` is the matrix that changes basis from:
        # the *canonical basis*: |up,up>, |up, down>, |down, up>, |down, down>
        # to the *two site basis*: |up,up>, sqrt(0.5) (|up, down> + |down, up>), sqrt(0.5) (|up, down> - |down, up>), |down, down>
        # Notice : we have P^T = P^-1 = P
        P = np.array([
            [1, 0, 0, 0],
            [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
            [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
            [0, 0, 0, 1],
        ])
        # The two site hamiltonian in its basis
        H_two_site = np.array([
            [self.problem.J_z / 4, 0, 0, 0],
            [0, (-self.problem.J_z / 4 + self.problem.J_x / 2), 0, 0],
            [0, 0, (-self.problem.J_z / 4 - self.problem.J_x / 2), 0],
            [0, 0, 0, self.problem.J_z / 4]
        ])
        exp_H_two_site = P @ np.diag(np.exp(np.diag(H_two_site))) @ P

        weight = 1

        # <\sigma_{\tau+1}| exp(-\delta\tau H_{1|2}) |\sigma_\tau>
        for tau in range(0, 2 * self.problem.m):
            tau_plus_one = tau + 1 % (2 * self.problem.m)

            # <\sigma_{2i, \tau+1},\sigma{2i+1, \tau+1}| exp(-\delta\tau H_two_site) |\sigma_{2i, \tau},\sigma{2i+1, \tau}>
            for i in range(0, self.problem.n_sites // 2):
                j = 2*i
                if tau % 2 == 1:
                    j = 2*i + 1

                bra = self.grid[tau_plus_one, j:j+2]  # <\sigma_{j, \tau+1},\sigma{j+1, \tau+1}|
                ket = self.grid[tau, j:j+2] # |\sigma_{j, \tau},\sigma{j+1, \tau}>
                weight *= bra.T @ exp_H_two_site @ ket

        return weight

    def draw(self):
        pass

