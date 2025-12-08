from .problem import Problem
import numpy as np
import matplotlib.pyplot as plt
import random


class Worldline:  # "w"
    """

    Attributes:
    - problem: The settings of this simulaton.
    - spins: The (2m, n) spins of the wordline: spins[i] is $\\ket{\\omega_i}$
    """

    def __init__(self, problem: Problem, spins=None):
        self.problem = problem
        self.spins = self._initialize_state() if spins is None else spins
        self.weight = self.compute_weight()
        if self.weight != 0:
            self.energy = self.compute_energy()
        else:
            self.energy = np.nan

    def copy(self):
        return Worldline(self.problem, self.spins.copy())

    def compute_weight(self):  # Omega(w)
        """
        Compute the weight $\\Omega(w)$ of the wordline configuration.
        """
        n = self.problem.n_sites
        m = self.problem.m
        weight = 1.0

        for tau in range(2 * m):
            for i in range(n // 2):
                spins_selected = (2 * i, (2 * i + 1) % n)
                if tau % 2 == 1:
                    spins_selected = ((2 * i + 1) % n, (2 * i + 2) % n)

                bra = list(self.spins[(tau + 1) % (2 * m), spins_selected])
                ket = list(self.spins[tau, spins_selected])
                # print(f"<{(tau+1) % (2*m)}, {spins_selected}|={bra}")
                # print(f"|{tau}, {spins_selected}>={ket}")

                if (ket == [1, -1] and bra == [1, -1]) or (
                    ket == [-1, 1] and bra == [-1, 1]
                ):
                    # print("a")
                    weight *= self.problem.weight_side
                elif (ket == [1, -1] and bra == [-1, 1]) or (
                    ket == [-1, 1] and bra == [1, -1]
                ):
                    # print("b")
                    weight *= self.problem.weight_cross
                elif (ket == [1, 1] and bra == [1, 1]) or (
                    ket == [-1, -1] and bra == [-1, -1]
                ):
                    # print("c")
                    weight *= self.problem.weight_full
                else:
                    weight = 0.0
                    break

            if weight == 0.0:
                break

        return weight

    def compute_energy(self):
        n = self.problem.n_sites
        m = self.problem.m
        energy = 0.0

        for tau in range(2 * m):
            for i in range(n // 2):
                spins_selected = (2 * i, (2 * i + 1) % n)
                if tau % 2 == 1:
                    spins_selected = ((2 * i + 1) % n, (2 * i + 2) % n)

                bra = list(self.spins[(tau + 1) % (2 * m), spins_selected])
                ket = list(self.spins[tau, spins_selected])

                if (ket == [1, -1] and bra == [1, -1]) or (
                    ket == [-1, 1] and bra == [-1, 1]
                ):
                    energy += self.problem.energy_side
                elif (ket == [1, -1] and bra == [-1, 1]) or (
                    ket == [-1, 1] and bra == [1, -1]
                ):
                    energy += self.problem.energy_cross
                elif (ket == [1, 1] and bra == [1, 1]) or (
                    ket == [-1, -1] and bra == [-1, -1]
                ):
                    energy += self.problem.energy_full
                else:
                    print("Warning: computing energy of an invalid wordline.")
                    return np.nan

        return energy / m

    def debug(self):
        rows, cols = self.spins.shape

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Iterate over grid to plot points
        # We invert the y-axis logic so row 0 is at the top (matrix style)
        for r in range(rows):
            for c in range(cols):
                val = self.spins[r, c]
                color = "red" if val == 1 else "black"

                # Plot point.
                # x = column index
                # y = inverted row index (rows - 1 - r)
                ax.scatter(c, rows - 1 - r, c=color, s=200, edgecolors="gray", zorder=2)

        # Set limits and grid
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect("equal")

        # Tick labels
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        # Invert y-labels so 0 is at the top
        ax.set_yticklabels(range(rows)[::-1])

        ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
        ax.set_title("Configuration (Red=1, Black=-1)")
        plt.xlabel("n_sites (columns)")
        plt.ylabel("2*m (rows)")

        plt.show()

    def draw(self, grid=None):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        if grid.all() == None:
            grid = self.spins

        n = self.problem.n_sites
        m = int(2 * self.problem.m)
        spin_color = {1: "red", -1: "cornflowerblue", 0: "black"}

        # grid = self.spins
        plt.figure()

        tiles = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                tiles[i, j] = (
                    j + i % 2 + 1
                ) % 2  # checkerboard pattern for better visibility

        # cmap='gray' maps -1->black, 1->white; interpolation='nearest' for sharp squares
        plt.imshow(tiles, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        pad = 0.05  # small offset inside each cell
        for i in range(m):
            for j in range(n):
                # place text at top-left of the cell with a small margin
                plt.text(
                    j - 0.5 + pad,
                    i - 0.5 + pad,
                    str(grid[i, j]),
                    color=spin_color[grid[i, j]],
                    ha="left",
                    va="top",
                )
            plt.text(
                n - 0.5 + pad,
                i - 0.5 + pad,
                str(grid[i, 0]),
                color=spin_color[grid[i, 0]],
                ha="left",
                va="top",
            )
        for j in range(n):
            # place text at top-left of the cell with a small margin
            plt.text(
                j - 0.5 + pad,
                m - 0.5 + pad,
                str(grid[0, j]),
                color=spin_color[grid[0, j]],
                ha="left",
                va="top",
            )
        plt.text(
            n - 0.5 + pad,
            m - 0.5 + pad,
            str(grid[0, 0]),
            color=spin_color[grid[0, 0]],
            ha="left",
            va="top",
        )

        # Plot the wordlines

        for i in range(m):
            for j in range(n // 2):
                if i % 2 == 0:
                    if (
                        grid[i, 2 * j % n] == grid[(i + 1) % m, 2 * j % n]
                        and grid[i, (2 * j + 1) % n]
                        == grid[(i + 1) % m, (2 * j + 1) % n]
                    ):
                        plt.plot(
                            [2 * j - 0.5, 2 * j - 0.5],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, 2 * j % n]],
                            linewidth=2,
                        )
                        plt.plot(
                            [2 * j + 0.5, 2 * j + 0.5],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 1) % n]],
                            linewidth=2,
                        )

                    if (
                        grid[i, 2 * j % n] != grid[i, (2 * j + 1) % n]
                        and grid[i, 2 * j % n] == grid[(i + 1) % m, (2 * j + 1) % n]
                        and grid[i, (2 * j + 1) % n] == grid[(i + 1) % m, 2 * j % n]
                    ):
                        plt.plot(
                            [2 * j - 0.5, 2 * j - 0.5 + 1],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j) % n]],
                            linewidth=2,
                        )
                        plt.plot(
                            [2 * j - 0.5 + 1, 2 * j - 0.5],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 1) % n]],
                            linewidth=2,
                        )

                if i % 2 == 1:
                    if (
                        grid[i, (2 * j + 1) % n] == grid[(i + 1) % m, (2 * j + 1) % n]
                        and grid[i, (2 * j + 2) % n]
                        == grid[(i + 1) % m, (2 * j + 2) % n]
                    ):
                        plt.plot(
                            [2 * j - 0.5 + 1, 2 * j - 0.5 + 1],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 1) % n]],
                            linewidth=2,
                        )
                        plt.plot(
                            [2 * j + 0.5 + 1, 2 * j + 0.5 + 1],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 2) % n]],
                            linewidth=2,
                        )

                    if (
                        grid[i, (2 * j + 1) % n] != grid[i, (2 * j + 2) % n]
                        and grid[i, (2 * j + 1) % n]
                        == grid[(i + 1) % m, (2 * j + 2) % n]
                        and grid[i, (2 * j + 2) % n]
                        == grid[(i + 1) % m, (2 * j + 1) % n]
                    ):
                        plt.plot(
                            [2 * j - 0.5 + 1, 2 * j - 0.5 + 2],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 1) % n]],
                            linewidth=2,
                        )
                        plt.plot(
                            [2 * j - 0.5 + 2, 2 * j - 0.5 + 1],
                            [i - 0.5, i + 0.5],
                            color=spin_color[grid[i, (2 * j + 2) % n]],
                            linewidth=2,
                        )

        # plt.xticks([]); plt.yticks([])
        plt.show()

    def _initialize_state(self):
        i, j0 = 0, 0
        set_spins = [0] * self.problem.n_sites
        self.spins, _, _ = self.generate_valid_state(self.spins, i, j0, set_spins)

    def generate_valid_state(self, spins, i, j0, set_spins):  # with prints
        m = int(2 * self.problem.m)
        n = self.problem.n_sites
        # print(f"drawing at i={i}, j0={j0}")
        # self.draw(spins)

        # If we're at the top row and the current cell is unset, choose a random spin value
        if i == 0:
            set_spins[j0] = 1

        if i == 0 and spins[i, j0] == 0:
            # print(f"i=0, j0 = {j0} unset spin")
            set_spins[j0] = 1
            chosen_value = random.choice([-1, 1])
            # print(f"i=0, j0 = {j0} unset spin, chose value ", chosen_value)
            new_spins = spins.copy()
            new_spins[i, j0] = chosen_value
            new_set_spins = set_spins.copy()
            new_spins, new_set_spins, validity = self.generate_valid_state(
                new_spins, i, j0, new_set_spins
            )
            if validity:
                # print(f"i=0, j0 = {j0} chosen value {chosen_value} is valid")
                return new_spins, new_set_spins, True
            else:
                # print(f"i=0, j0 = {j0} chosen value {chosen_value} is not valid, new value is {-chosen_value}")
                chosen_value = -chosen_value
                spins[0, j0] = chosen_value
                return self.generate_valid_state(spins, i, j0, set_spins)

        # print(f"i= {i}, j0 = {j0}, testing neighbor")
        k = 1 - 2 * ((i + j0) % 2)  # k = 1 if same parity, -1 if different parity
        neighbor = (j0 + k) % n

        parity = (
            spins[i, j0] * spins[i, neighbor]
        )  # 1 if both have the same spin, -1 if opposite spins, 0 if neighbor is unset

        if i == m - 1:
            # print(f"i= {i}, j0 = {j0}, at last row")
            if spins[(i + 1) % m, j0] != 0 and spins[(i + 1) % m, neighbor] != 0:
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins set")
                if (
                    spins[(i + 1) % m, j0]
                    * spins[i, j0]
                    * spins[(i + 1) % m, neighbor]
                    * spins[i, neighbor]
                    == -1
                ):
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, incompatible spins")
                    return spins, set_spins, False

                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, looking for new j")
                    j = 0
                    while set_spins[j % n] == 1 and j < n:
                        j += 1
                    if set_spins[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, found new j = {j}")
                        return self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins
                        )
                    else:
                        # print("all spins are set, returning valid state")
                        return spins, set_spins, True

            if spins[(i + 1) % m, j0] == 0 and spins[(i + 1) % m, neighbor] == 0:
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset")
                chosen_spin = random.choice([j0, neighbor])
                new_spins = spins.copy()
                new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                new_set_spins = set_spins.copy()
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin {chosen_spin}")

                if new_set_spins[chosen_spin] == 0:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin")
                    new_set_spins[chosen_spin] = 1
                    new_spins, new_set_spins, validity = self.generate_valid_state(
                        new_spins, (i + 1) % m, chosen_spin, new_set_spins
                    )

                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j")
                    j = 0
                    while new_set_spins[j % n] == 1 and j < n:
                        j += 1
                    if new_set_spins[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}")
                        new_spins, new_set_spins, validity = self.generate_valid_state(
                            new_spins, (i + 1) % m, j, new_set_spins
                        )
                    else:
                        # print("all spins are set, returning valid state")
                        return new_spins, new_set_spins, True

                if validity:
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin {chosen_spin}, valid")
                    return new_spins, new_set_spins, True
                else:
                    chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin  not valid, chosen spin {chosen_spin} instead")
                    spins[(i + 1) % m, chosen_spin] = spins[i, j0]

            elif spins[(i + 1) % m, j0] != 0 or spins[(i + 1) % m, neighbor] != 0:
                # print(f"i= {i}, j0 = {j0}, at last row, one below spin set")
                if parity == 1:
                    # print(f"i= {i}, j0 = {j0}, at last row, parity=1")
                    if spins[(i + 1) % m, j0] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=1, j0 unset, setting spin at j0")
                        spins[(i + 1) % m, j0] = spins[i, j0]
                        chosen_spin = j0

                    else:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=1, j0 set, invalid state")
                        return spins, set_spins, False

                if parity == -1:
                    chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
                    # print(f"i= {i}, j0 = {j0}, at last row, parity=-1, setting spin at {chosen_spin} as its the one empty")
                    spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                if parity == 0:
                    # print(f"i= {i}, j0 = {j0}, at last row, parity=0, checking if the set spin is compatible")
                    j1 = j0 if spins[(i + 1) % m, j0] != 0 else neighbor
                    if spins[(i + 1) % m, j1] != spins[i, j0]:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is opposite, its a valid state, setting the other spin")
                        chosen_spin = j0 if j1 != j0 else neighbor
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    elif spins[(i + 1) % m, j1] == spins[i, j0] and j1 == j0:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is same and below, only one option as we cannot cross with same spins")
                        chosen_spin = j0
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    else:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is same, need to try both options, going to the set spin or to the other as both are possible")
                        chosen_spin = random.choice([j0, neighbor])
                        new_spins = spins.copy()
                        new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                        new_set_spins = set_spins.copy()
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=0, trying chosen spin {chosen_spin}")

                        if new_set_spins[chosen_spin] == 0:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin")
                            new_set_spins[chosen_spin] = 1
                            new_spins, new_set_spins, validity = (
                                self.generate_valid_state(
                                    new_spins, (i + 1) % m, chosen_spin, new_set_spins
                                )
                            )

                        else:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j")
                            j = 0
                            while new_set_spins[j % n] == 1 and j < n:
                                j += 1
                            if new_set_spins[j % n] == 0:
                                # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}")
                                new_spins, new_set_spins, validity = (
                                    self.generate_valid_state(
                                        new_spins, (i + 1) % m, j, new_set_spins
                                    )
                                )
                            else:
                                # print("all spins are set, returning valid state")
                                return new_spins, new_set_spins, True

                        if validity:
                            # print(f"i= {i}, j0 = {j0}, at last row, parity=0, chosen spin {chosen_spin} is valid")
                            return new_spins, new_set_spins, True
                        else:
                            # print(f"i= {i}, j0 = {j0}, at last row, parity=0, chosen spin {chosen_spin} is not valid, trying the other spin")
                            chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                            spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} now seeing if it has already been set  before")
                if set_spins[chosen_spin] == 0:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin")
                    set_spins[chosen_spin] = 1
                    return self.generate_valid_state(
                        spins, (i + 1) % m, chosen_spin, set_spins
                    )

                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j")
                    j = 0
                    while set_spins[j % n] == 1 and j < n:
                        j += 1
                    if set_spins[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}")
                        return self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins
                        )
                    else:
                        # print("all spins are set, returning valid state")
                        return spins, set_spins, True

        if parity == 0:
            # print(f"i= {i}, j0 = {j0}, parity=0")
            chosen_spin = random.choice([j0, neighbor])
            new_spins = spins.copy()
            new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            new_set_spins = set_spins.copy()
            # print(f"i= {i}, j0 = {j0}, parity=0, trying chosen spin {chosen_spin}")
            new_spins, new_set_spins, validity = self.generate_valid_state(
                new_spins, (i + 1) % m, chosen_spin, new_set_spins
            )
            if validity:
                # print(f"i= {i}, j0 = {j0}, parity=0, chosen spin {chosen_spin} is valid")
                return new_spins, new_set_spins, True
            else:
                # print(f"i= {i}, j0 = {j0}, parity=0, chosen spin {chosen_spin} is not valid, trying the other spin")
                chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                return self.generate_valid_state(
                    spins, (i + 1) % m, chosen_spin, set_spins
                )

        if parity == 1:
            # print(f"i= {i}, j0 = {j0}, parity=1")
            if spins[(i + 1) % m, j0] == 0:
                # print(f"i= {i}, j0 = {j0}, parity=1, j0 unset, setting spin at j0")
                spins[(i + 1) % m, j0] = spins[i, j0]
                return self.generate_valid_state(spins, (i + 1) % m, j0, set_spins)

            else:
                # print(f"i= {i}, j0 = {j0}, parity=1, j0 set, invalid state")
                return spins, set_spins, False

        if parity == -1:
            # print(f"i= {i}, j0 = {j0}, parity=-1, going to set the only available spin")
            chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
            spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            return self.generate_valid_state(spins, (i + 1) % m, chosen_spin, set_spins)

        # print(" WARNING Should not reach here")


    
class ExhaustiveWorldline:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.worldlines = self.generate_all_wordlines()

    def generate_all_wordlines(self):
        # ensure we start with an empty result list on each call
        worldlines = []
        # set to record seen configurations (avoid duplicates)
        spins = np.zeros((int(2*self.problem.m), self.problem.n_sites), dtype=int)
        set_spins = [0]*self.problem.n_sites
        i, j0 = 0, 0
        self.generate_valid_state(spins, i, j0, set_spins, worldlines)
        return worldlines

    

    def generate_valid_state(self, spins, i, j0, set_spins, worldlines):
        m = int(2*self.problem.m)
        n = self.problem.n_sites
        # self.draw(spins)

        # If we're at the top row and the current cell is unset, choose a random spin value
        if i == 0:
            set_spins[j0] = 1
            # At top row: if unset, branch both spin choices; otherwise mark column as seen and continue
            if spins[i, j0] == 0:
                new_spins1 = spins.copy()
                new_spins1[i, j0] = 1
                new_set_spins1 = set_spins.copy()
                new_spins2 = spins.copy()
                new_spins2[i, j0] = -1
                new_set_spins2 = set_spins.copy()
                
            
                # Recursively continue processing to handle neighbor and propagation
                self.generate_valid_state(new_spins1, i, j0, new_set_spins1, worldlines)
                self.generate_valid_state(new_spins2, i, j0, new_set_spins2, worldlines)
                return
                

        k = 1 - 2 * ((i + j0) % 2)  # k = 1 if same parity, -1 if different parity
        neighbor = (j0 + k) % n

        a = spins[i, j0]*spins[i, neighbor] # 1 if both have the same spin, -1 if opposite spins, 0 if neighbor is unset


        if i == m-1:
            # print("i=m-1")
            if spins[(i+1)%m, j0] !=0 and spins[(i+1)%m, neighbor] !=0:
                if spins[(i+1)%m, j0]*spins[i, j0]*spins[(i+1)%m, neighbor]*spins[(i+1)%m, neighbor] == 1:
                    # print("i=m-1, on cherche un nouveau j")
                    j = 0
                    while j < n and set_spins[j] == 1:
                        j += 1
                    if j == n:
                        worldlines.append(spins.copy())
                        
                    else:
                        self.generate_valid_state(spins, (i+1)%m, j, set_spins, worldlines)
                
                return
                    
            if spins[(i+1)%m, j0] ==0 and spins[(i+1)%m, neighbor] ==0:
                # Try both downward placements
                new_spins1 = spins.copy()
                new_spins1[(i + 1)%m, j0] = spins[i, j0]
                new_set_spins1 = set_spins.copy()
                self.generate_valid_state(new_spins1, (i+1)%m, j0, new_set_spins1, worldlines)

                new_spins2 = spins.copy()
                new_spins2[(i + 1)%m, neighbor] = spins[i, j0]
                new_set_spins2 = set_spins.copy()
                self.generate_valid_state(new_spins2, (i+1)%m, neighbor, new_set_spins2, worldlines)
                return
                
            elif spins[(i+1)%m, j0] !=0 or spins[(i+1)%m, neighbor] !=0:
                if a == 1:
                    # both columns have same spin: set the row below for both columns if needed
                    if spins[(i + 1)%m, j0] == 0:
                        spins[(i + 1)%m, j0] = spins[i, j0]
                        chosen_spin = j0 
                    else:
                        return
                
                if a == -1:
                    if spins[(i + 1)%m, j0] == 0:
                        # print("a = -1, j0")
                        spins[(i + 1)%m, j0] = spins[i, j0]
                        chosen_spin = j0
                    
                    elif spins[(i + 1)%m, neighbor] == 0:
                        # print("a = -1, neighbor")
                        spins[(i + 1)%m, neighbor] = spins[i, j0]
                        chosen_spin = neighbor
                    
                    else:
                        # print("a = -1, refusé")
                        return

                if a == 0:
                    # one of the below row cells may already be set and possibly incompatible
                    j1 = j0 if spins[(i+1)%m, j0] !=0 else neighbor
                    if spins[(i+1)%m, j1] != spins[i, j0]:
                        chosen_spin = j0 if j1 != j0 else neighbor
                        spins[(i + 1)%m, chosen_spin] = spins[i, j0]
                    
                    else:
                        chosen_spin = j1
                        spins[(i + 1)%m, chosen_spin] = spins[i, j0]

                    # else:
                    #     # print("i=m-1, a=0, on essaye les deux choix")
                    #     chosen_spin = random.choice([j0, neighbor])
                    #     new_spins = spins.copy()
                    #     new_spins[(i + 1)%m, chosen_spin] = spins[i, j0]
                    #     new_set_spins = set_spins.copy()
                    #     new_set_spins[chosen_spin] = 1
                    #     new_spins, new_set_spins, validity = self.generate_valid_state(new_spins, (i+1)%m, chosen_spin, new_set_spins)
                    #     if validity:
                    #         # print("i=m-1, 0, 0, le changement est valide")
                    #         spins = new_spins
                    #         set_spins = new_set_spins
                    #         return spins, set_spins, True
                    #     else:
                    #         # print("i=m-1, 0, 0, le changement est pas valide")
                    #         chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                    #         spins[(i + 1)%m, chosen_spin] = spins[i, j0]
                    #         set_spins[chosen_spin] = 1
                    #         return self.generate_valid_state(spins, (i+1)%m, chosen_spin, set_spins)

                if set_spins[chosen_spin] == 0:
                    set_spins[chosen_spin] = 1
                    self.generate_valid_state(spins, (i+1)%m, chosen_spin, set_spins, worldlines)
                    return
                
                else:
                    j = 0
                    while j < n and set_spins[j] == 1:
                        j += 1
                    if j == n:
                        worldlines.append(spins.copy())
                    else:
                        self.generate_valid_state(spins, (i+1)%m, j, set_spins, worldlines)
                    return
                   
                    


        if a == 0:
            # try both downward continuations for exhaustive enumeration
            new_spins1 = spins.copy()
            new_spins1[(i + 1)%m, j0] = spins[i, j0]
            new_set_spins1 = set_spins.copy()
            self.generate_valid_state(new_spins1, (i+1)%m, j0, new_set_spins1, worldlines)

            new_spins2 = spins.copy()
            new_spins2[(i + 1)%m, neighbor] = spins[i, j0]
            new_set_spins2 = set_spins.copy()
            self.generate_valid_state(new_spins2, (i+1)%m, neighbor, new_set_spins2, worldlines)
            return
            # if validity:
            #     # print("i=m-1, 0, 0, le changement est valide")
            #     return new_spins, new_set_spins, True

                
        if a == 1:
            # straight continuation: set downward cell and recurse if the two cells do not cross
            if spins[(i + 1)%m, j0] == 0:
                spins[(i + 1)%m, j0] = spins[i, j0]
                self.generate_valid_state(spins, (i + 1)%m, j0, set_spins, worldlines)
            return
        
        if a == -1:
            # diagonal continuation required: check if its possible
            if spins[(i + 1)%m, j0] == 0:
                spins[(i + 1)%m, j0] = spins[i, j0]
                self.generate_valid_state(spins, (i + 1)%m, j0, set_spins, worldlines)

            elif spins[(i + 1)%m, neighbor] == 0:
                spins[(i + 1)%m, neighbor] = spins[i, j0]
                self.generate_valid_state(spins, (i + 1)%m, neighbor, set_spins, worldlines)

            return
        
        return

        
    
    def draw_worldline(self, grid):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        n = self.problem.n_sites
        m = int(2*self.problem.m)
        spin_color = {1: "red", -1: "cornflowerblue", 0: "black"}

        
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
