from production.problem import Problem
import numpy as np
import matplotlib.pyplot as plt
import functools


class Wordline:  # "w"
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
        return Wordline(self.problem, self.spins.copy())

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

    def draw(self):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        n = self.problem.n_sites
        m = 2 * self.problem.m
        spin_color = {1: "red", -1: "cornflowerblue"}

        grid = self.spins
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
        n = self.problem.n_sites
        m = 2 * self.problem.m
        # Initialize a random state for the wordline
        grid = np.zeros((m, n), dtype=int)
        grid[0, :] = np.random.choice([-1, 1], size=(n))

        # Probablity of exchange
        position_list = {}
        probability_list = np.zeros(m)
        for i in range(m):
            probability_list[i] = 0.5
        for i in range(n):
            position_list[i] = i

        for i in range(m - 1):
            for j in range(n // 2):
                if i % 2 == 0:
                    grid[(i + 1) % m, (2 * j) % n] = grid[(i) % m, (2 * j) % n]
                    grid[(i + 1) % m, (2 * j + 1) % n] = grid[(i) % m, (2 * j + 1) % n]
                    if grid[(i) % m, (2 * j) % n] * grid[
                        (i) % m, (2 * j + 1) % n
                    ] == -1 and np.random.rand() < min(
                        probability_list[(2 * j) % n],
                        1 - probability_list[(2 * j + 1) % n],
                    ):
                        grid[(i + 1) % m, (2 * j) % n] = -grid[(i) % m, (2 * j) % n]
                        grid[(i + 1) % m, (2 * j + 1) % n] = -grid[
                            (i) % m, (2 * j + 1) % n
                        ]

                        # Update positions
                        temp_position = position_list[(2 * j) % n]
                        position_list[(2 * j) % n] = position_list[(2 * j + 1) % n]
                        position_list[(2 * j + 1) % n] = temp_position

                        # Update probabilities
                        d = (2 * j) % n - position_list[(2 * j) % n]
                        if d != 0:
                            p = (m - i - 1) / min(
                                abs(d), n - abs(d)
                            )  # probabilité d'échanger vers la droite
                        if d == 0:
                            probability_list[(2 * j) % n] = 0.5
                            if i >= m - 2:
                                grid[(i + 1) % m, (2 * j) % n] = -grid[
                                    (i) % m, (2 * j) % n
                                ]
                                grid[(i + 1) % m, (2 * j + 1) % n] = -grid[
                                    (i) % m, (2 * j + 1) % n
                                ]

                        elif (abs(d) < n - abs(d) and d > 0) or (
                            d < 0 and abs(d) > n - abs(d)
                        ):
                            probability_list[(2 * j) % n] = 1 - p
                        else:
                            probability_list[(2 * j) % n] = p

                        d = (2 * j + 1) % n - position_list[(2 * j + 1) % n]
                        if d == 0:
                            probability_list[(2 * j + 1) % n] = 0.5
                        elif (abs(d) < n - abs(d) and d > 0) or (
                            d < 0 and abs(d) > n - abs(d)
                        ):
                            probability_list[(2 * j + 1) % n] = 1 - p
                        else:
                            probability_list[(2 * j + 1) % n] = p

                if i % 2 == 1:
                    grid[(i + 1) % m, (2 * j + 2) % n] = grid[(i) % m, (2 * j + 2) % n]
                    grid[(i + 1) % m, (2 * j + 1) % n] = grid[(i) % m, (2 * j + 1) % n]
                    if grid[(i) % m, (2 * j + 2) % n] * grid[
                        (i) % m, (2 * j + 1) % n
                    ] == -1 and np.random.rand() < min(
                        1 - probability_list[(2 * j + 2) % n],
                        probability_list[(2 * j + 1) % n],
                    ):
                        grid[(i + 1) % m, (2 * j + 2) % n] = -grid[
                            (i) % m, (2 * j + 2) % n
                        ]
                        grid[(i + 1) % m, (2 * j + 1) % n] = -grid[
                            (i) % m, (2 * j + 1) % n
                        ]

                        # Update positions
                        temp_position = position_list[(2 * j + 2) % n]
                        position_list[(2 * j + 2) % n] = position_list[(2 * j + 1) % n]
                        position_list[(2 * j + 1) % n] = temp_position

                        # Update probabilities
                        d = (2 * j + 2) % n - position_list[(2 * j + 2) % n]
                        if d != 0:
                            p = (m - i - 1) / min(
                                abs(d), n - abs(d)
                            )  # probabilité d'échanger vers la droite
                        if d == 0:
                            probability_list[(2 * j + 2) % n] = 0.5
                            if i >= m - 2:
                                grid[(i + 1) % m, (2 * j) % n] = -grid[
                                    (i) % m, (2 * j) % n
                                ]
                                grid[(i + 1) % m, (2 * j + 1) % n] = -grid[
                                    (i) % m, (2 * j + 1) % n
                                ]
                        elif (abs(d) < n - abs(d) and d > 0) or (
                            d < 0 and abs(d) > n - abs(d)
                        ):
                            probability_list[(2 * j + 2) % n] = 1 - p
                        else:
                            probability_list[(2 * j + 2) % n] = p

                        d = (2 * j + 1) % n - position_list[(2 * j + 1) % n]
                        if d == 0:
                            probability_list[(2 * j + 1) % n] = 0.5
                        elif (abs(d) < n - abs(d) and d > 0) or (
                            d < 0 and abs(d) > n - abs(d)
                        ):
                            probability_list[(2 * j + 1) % n] = 1 - p
                        else:
                            probability_list[(2 * j + 1) % n] = p

        return grid


class ExhaustiveWordline(Wordline):
    def __init__(self, problem: Problem, grid: np.ndarray):
        super().__init__(problem)
        self.grid = grid

    def weight(self):  # Omega(w)
        # `P` is the matrix that changes basis from:
        # the *canonical basis*: |up,up>, |up, down>, |down, up>, |down, down>
        # to the *two site basis*: |up,up>, sqrt(0.5) (|up, down> + |down, up>), sqrt(0.5) (|up, down> - |down, up>), |down, down>
        # Notice : we have P^T = P^-1 = P
        P = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        # The two site hamiltonian in its basis
        H_two_site = np.array(
            [
                [self.problem.J_z / 4, 0, 0, 0],
                [0, (-self.problem.J_z / 4 + self.problem.J_x / 2), 0, 0],
                [0, 0, (-self.problem.J_z / 4 - self.problem.J_x / 2), 0],
                [0, 0, 0, self.problem.J_z / 4],
            ]
        )
        exp_H_two_site = P @ np.diag(np.exp(np.diag(H_two_site))) @ P

        weight = 1

        # <\sigma_{\tau+1}| exp(-\delta\tau H_{1|2}) |\sigma_\tau>
        for tau in range(0, 2 * self.problem.m):
            tau_plus_one = tau + 1 % (2 * self.problem.m)

            # <\sigma_{2i, \tau+1},\sigma{2i+1, \tau+1}| exp(-\delta\tau H_two_site) |\sigma_{2i, \tau},\sigma{2i+1, \tau}>
            for i in range(0, self.problem.n_sites // 2):
                j = 2 * i
                if tau % 2 == 1:
                    j = 2 * i + 1

                bra = self.grid[
                    tau_plus_one, j : j + 2
                ]  # <\sigma_{j, \tau+1},\sigma{j+1, \tau+1}|
                ket = self.grid[tau, j : j + 2]  # |\sigma_{j, \tau},\sigma{j+1, \tau}>
                weight *= bra.T @ exp_H_two_site @ ket

        return weight
