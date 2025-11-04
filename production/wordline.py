from problem import Problem
import numpy as np
import random
import matplotlib.pyplot as plt

class Wordline: # "w"
    def __init__(self, problem: Problem):
        self.problem = problem
        self.grid = self.initialize_state()

    def weight(self): # Omega(w)
        pass

    def draw(self):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        n = self.problem.n_sites
        m = self.problem.m

        grid = self.grid
        plt.figure()

        # cmap='gray' maps -1->black, 1->white; interpolation='nearest' for sharp squares
        plt.imshow(grid, cmap="gray", interpolation="nearest", vmin=-1, vmax=1)
        for i in range(m):
            for j in range(n):
                plt.text(j, i, str(grid[i, j]), ha="center", va="center", color="red")

        plt.xticks([]); plt.yticks([])
        plt.gca().set_aspect("equal")
        plt.show()
        
    
    def initialize_state(self):
        n = self.problem.n_sites
        m = self.problem.m
        # Initialize a random state for the wordline
        grid = np.zeros((m, n), dtype=int)
        grid[0, :] = np.random.choice([-1, 1], size=(n))
        for i in range(1, m):
            for j in range(n//2):
                if m%2 == 0:
                    if grid[i, 2*j]*grid[i, 2*j+1] == -1 and np.random.rand() < 0.5:
                        grid[i, 2*j] *= -1
                        grid[i, 2*j+1] *= -1
                
                if m%2 == 1:
                    if grid[i, 2*j+2]*grid[i, 2*j+1] == -1 and np.random.rand() < 0.5:
                        grid[i, 2*j+2] *= -1
                        grid[i, 2*j+1] *= -1
                
                grid[i, 0] = grid[i, n-1]
        return grid



