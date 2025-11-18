
from production.problem import Problem
import numpy as np
import random
import matplotlib.pyplot as plt

class Wordline: # "w"
    """
    
    Attributes:
    - problem: The settings of this simulaton.
    - spins: The (2m, n) spins of the wordline: spins[i] is $\ket{\omega_i}$
    """
    def __init__(self, problem: Problem, spins = None):
        self.problem = problem
        self.spins = self._initialize_state() if spins is None else spins
        self._weight = None

    @property
    def weight(self):
        if self._weight == None:
            self._weight = self._compute_weight()
        return self._weight

    def _compute_weight(self): # Omega(w)
        """
        Compute the weight $\Omega(w)$ of the wordline configuration.
        """
        n = self.problem.n_sites
        m = self.problem.m
        weight = 1.0

        for tau in range(2*m):
            for i in range(n//2):
                spins_selected = (2*i, (2*i+1) % n)
                if tau % 2 == 1:
                    spins_selected = ((2*i+1)%n, (2*i+2) % n)
                
                bra = list(self.spins[(tau + 1) % (2*m), spins_selected])
                ket = list(self.spins[tau, spins_selected])
                # print(f"<{(tau+1) % (2*m)}, {spins_selected}|={bra}")
                # print(f"|{tau}, {spins_selected}>={ket}")

                if (ket == [1, -1] and bra == [1, -1]) or (ket == [-1, 1] and bra == [-1, 1]):
                    # print("a")
                    weight *= np.exp(self.problem.delta_tau * self.problem.J_z / 4) * np.cosh(self.problem.delta_tau * self.problem.J_x / 2)
                elif (ket == [1, -1] and bra == [-1, 1]) or (ket == [-1, 1] and bra == [1, -1]):
                    # print("b")
                    weight *= -np.exp(self.problem.delta_tau * self.problem.J_z / 4) * np.sinh(self.problem.delta_tau * self.problem.J_x / 2)
                elif (ket == [1, 1] and bra == [1, 1]) or (ket == [-1, -1] and bra == [-1, -1]):
                    # print("c")
                    weight *= np.exp(-self.problem.delta_tau * self.problem.J_z / 4)
                else:
                    weight = 0.0
                    break

            if weight == 0.0:
                break

        return weight

    def draw(self):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        n = self.problem.n_sites
        m = 2*self.problem.m
        spin_color = {1: "red", -1: "cornflowerblue"}

        grid = self.spins
        plt.figure()

        tiles = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                tiles[i, j] = (j+i%2+1)%2 # checkerboard pattern for better visibility


        # cmap='gray' maps -1->black, 1->white; interpolation='nearest' for sharp squares
        plt.imshow(tiles, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
         
        pad = 0.05  # small offset inside each cell
        for i in range(m):
            for j in range(n):
                # place text at top-left of the cell with a small margin
                plt.text(j - 0.5 + pad, i - 0.5 + pad, str(grid[i, j]),
                        color=spin_color[grid[i, j]], ha="left", va="top")
            plt.text(n - 0.5 + pad, i - 0.5 + pad, str(grid[i, 0]),
                        color=spin_color[grid[i, 0]], ha="left", va="top")
        for j in range(n):
                # place text at top-left of the cell with a small margin
                plt.text(j - 0.5 + pad, m - 0.5 + pad, str(grid[0, j]),
                        color=spin_color[grid[0, j]], ha="left", va="top")
        plt.text(n - 0.5 + pad, m - 0.5 + pad, str(grid[0, 0]),
                        color=spin_color[grid[0, 0]], ha="left", va="top")
        

        #Plot the wordlines
        

        for i in range(m):
            for j in range(n//2):
                 

                if i%2 == 0:
                    if grid[i, 2*j%n] == grid[(i+1)%m, 2*j%n] and grid[i, (2*j+1)%n] == grid[(i+1)%m, (2*j+1)%n]:
                        plt.plot([2*j-0.5, 2*j-0.5], [i-0.5, i+0.5], color=spin_color[grid[i, 2*j%n]], linewidth=2)
                        plt.plot([2*j+0.5, 2*j+0.5], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+1)%n]], linewidth=2)

                    if grid[i, 2*j%n] != grid[i, (2*j+1)%n] and grid[i, 2*j%n] == grid[(i+1)%m, (2*j+1)%n] and grid[i, (2*j+1)%n] == grid[(i+1)%m, 2*j%n]:
                        plt.plot([2*j-0.5, 2*j-0.5 +1], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j)%n]], linewidth=2)
                        plt.plot([2*j-0.5+1, 2*j-0.5], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+1)%n]], linewidth=2)
                
                if i%2 == 1 :
                    if grid[i, (2*j+1)%n] == grid[(i+1)%m, (2*j+1)%n] and grid[i, (2*j+2)%n] == grid[(i+1)%m, (2*j+2)%n]:
                        plt.plot([2*j-0.5+1, 2*j-0.5+1], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+1)%n]], linewidth=2)
                        plt.plot([2*j+0.5+1, 2*j+0.5+1], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+2)%n]], linewidth=2)
                    
                    if grid[i, (2*j+1)%n] != grid[i, (2*j+2)%n] and grid[i, (2*j+1)%n] == grid[(i+1)%m, (2*j+2)%n] and grid[i, (2*j+2)%n] == grid[(i+1)%m, (2*j+1)%n]:
                        plt.plot([2*j-0.5+1, 2*j-0.5 +2], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+1)%n]], linewidth=2)
                        plt.plot([2*j-0.5+2, 2*j-0.5+1], [i-0.5, i+0.5], color=spin_color[grid[i, (2*j+2)%n]], linewidth=2)
                    

        # plt.xticks([]); plt.yticks([])
        plt.show()
        
    
    def _initialize_state(self):
        n = self.problem.n_sites
        m = self.problem.m
        # Initialize a random state for the wordline
        grid = np.zeros((m, n), dtype=int)
        grid[0, :] = np.random.choice([-1, 1], size=(n))

        # Probablity of exchange 
        position_list = {}
        probability_list = np.zeros(m)
        for i in range(m):
            probability_list[i] = 0.5
        for i in range (n):
            position_list[i] = i
        
        for i in range(m-1):
            for j in range(n//2):
                if i%2 == 0:
                    grid[(i+1)%m, (2*j)%n] = grid[(i)%m, (2*j)%n]
                    grid[(i+1)%m, (2*j+1)%n] = grid[(i)%m, (2*j+1)%n]
                    if grid[(i)%m, (2*j)%n]*grid[(i)%m, (2*j+1)%n] == -1 and np.random.rand() < min(probability_list[(2*j)%n], 1 - probability_list[(2*j+1)%n]):
                        grid[(i+1)%m, (2*j)%n] = -grid[(i)%m, (2*j)%n]
                        grid[(i+1)%m, (2*j+1)%n] = -grid[(i)%m, (2*j+1)%n]


                        # Update positions 
                        temp_position = position_list[(2*j)%n]
                        position_list[(2*j)%n] = position_list[(2*j+1)%n]
                        position_list[(2*j+1)%n] = temp_position

                        # Update probabilities
                        d = (2*j)%n - position_list[(2*j)%n]
                        if d!= 0:  p = (m-i-1)/min(abs(d), n-abs(d)) # probabilité d'échanger vers la droite
                        if d == 0:
                            probability_list[(2*j)%n] = 0.5
                            if i >= m-2:
                                grid[(i+1)%m, (2*j)%n] = -grid[(i)%m, (2*j)%n]
                                grid[(i+1)%m, (2*j+1)%n] = -grid[(i)%m, (2*j+1)%n]


                        elif (abs(d) <  n - abs(d) and d>0) or (d<0 and abs(d) > n - abs(d)):
                            probability_list[(2*j)%n] = 1 -p
                        else:
                            probability_list[(2*j)%n] = p

                        d = (2*j+1)%n - position_list[(2*j+1)%n]
                        if d == 0: 
                            probability_list[(2*j+1)%n] = 0.5
                        elif (abs(d) <  n - abs(d) and d>0) or (d<0 and abs(d) > n - abs(d)):
                            probability_list[(2*j+1)%n] = 1 - p
                        else:
                            probability_list[(2*j+1)%n] = p



                
                if i%2 == 1:
                    grid[(i+1)%m, (2*j+2)%n] = grid[(i)%m, (2*j+2)%n]
                    grid[(i+1)%m, (2*j+1)%n] = grid[(i)%m, (2*j+1)%n]
                    if grid[(i)%m, (2*j+2)%n]*grid[(i)%m, (2*j+1)%n] == -1 and np.random.rand() < min(1 - probability_list[(2*j+2)%n], probability_list[(2*j+1)%n]):
                        grid[(i+1)%m, (2*j+2)%n] = -grid[(i)%m, (2*j+2)%n]
                        grid[(i+1)%m, (2*j+1)%n] = -grid[(i)%m, (2*j+1)%n]

                        # Update positions
                        temp_position = position_list[(2*j+2)%n]
                        position_list[(2*j+2)%n] = position_list[(2*j+1)%n]
                        position_list[(2*j+1)%n] = temp_position

                        # Update probabilities
                        d = (2*j+2)%n - position_list[(2*j+2)%n]
                        if d!= 0:  p = (m-i-1)/min(abs(d), n-abs(d)) # probabilité d'échanger vers la droite
                        if d == 0:
                            probability_list[(2*j+2)%n] = 0.5
                            if i >= m-2:
                                grid[(i+1)%m, (2*j)%n] = -grid[(i)%m, (2*j)%n]
                                grid[(i+1)%m, (2*j+1)%n] = -grid[(i)%m, (2*j+1)%n]
                        elif (abs(d) <  n - abs(d) and d>0) or (d<0 and abs(d) > n - abs(d)):
                            probability_list[(2*j+2)%n] = 1 - p
                        else:
                            probability_list[(2*j+2)%n] = p
                        
                        d = (2*j+1)%n - position_list[(2*j+1)%n]
                        if d == 0: 
                            probability_list[(2*j+1)%n] = 0.5
                        elif (abs(d) <  n - abs(d) and d>0) or (d<0 and abs(d) > n - abs(d)):
                            probability_list[(2*j+1)%n] = 1 - p
                        else:
                            probability_list[(2*j+1)%n] =  p
                
             
            

        return grid
    
class ExhaustiveWordline(Wordline):
    def __init__(self, problem: Problem, grid: np.ndarray):
        super().__init__(problem)
        self.grid = grid

    def weight(self): # Omega(w)
        pass



