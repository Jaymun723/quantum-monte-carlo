from .problem import Problem
import numpy as np
import matplotlib.pyplot as plt
import random


class Worldline:  # "w"
    """

    Attributes:
    - problem: The settings of this simulaton.
    - spins: The (2m, n) spins of the wordline: spins[i] is $\\ket{\\omega_i}$
    - weight: The weight of the Worldline.
    - energy: Energy attributed to the Worldline. NaN is weight is 0.
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

    def draw(self, grid=None):
        m = self.problem.m
        n = self.problem.n_sites

        plt.xticks(range(n))
        plt.yticks(range(2 * m))
        # plt.grid()
        plt.xlim(0, n)
        plt.ylim(0, 2 * m)

        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("sites ($n$)")
        ax.set_ylabel("imaginary time ($2m$)")

        spins = self.spins
        if grid is not None:
            spins = grid
            m = len(grid) // 2
            n = len(grid[0])

        tiles = np.zeros((2 * m, n))
        for j in range(2 * m):
            for i in range(n):
                tiles[j, i] = (
                    i + j % 2 + 1
                ) % 2  # checkerboard pattern for better visibility

        plt.imshow(
            tiles,
            cmap="gray",
            interpolation="nearest",
            vmin=-0.1,
            vmax=1,
            extent=(0, n, 2 * m, 0),
        )

        spin_color = {1: "red", -1: "cornflowerblue", 0: "black"}

        for i in range((2 * m) + 1):
            for j in range(n + 1):
                plt.plot(
                    j,
                    i,
                    marker="o",
                    color=spin_color[spins[i % (2 * m), j % n]],
                    markersize=5,
                )

        for i in range((2 * m)):
            for j in range(1, n):
                k = 1 - 2 * (
                    (i % (2 * m) + j) % 2
                )  # k = 1 if same parity, -1 if different parity
                neighbor = (j + k) % n
                i1, j1 = (i + 1) % (2 * m + 1), (j + k) % (n + 1)
                up = (i + 1) % (2 * m)

                c = spins[i % (2 * m), j % n] * spins[up, j % n]
                d = spins[i % (2 * m), neighbor] * spins[up, neighbor]

                if c == 1 and d == 1:
                    # straight lines
                    plt.plot(
                        [j, j],
                        [i, i1],
                        color=spin_color[spins[i % (2 * m), j % n]],
                        linewidth=2,
                    )
                    plt.plot(
                        [j1, j1],
                        [i, i1],
                        color=spin_color[spins[i % (2 * m), neighbor]],
                        linewidth=2,
                    )

                if c == -1 and d == -1:
                    # crossed lines
                    plt.plot(
                        [j, j1],
                        [i, i1],
                        color=spin_color[spins[i % (2 * m), j % n]],
                        linewidth=2,
                    )
                    plt.plot(
                        [j1, j],
                        [i, i1],
                        color=spin_color[spins[i % (2 * m), neighbor]],
                        linewidth=2,
                    )

                if c * d == -1:
                    # Impossible square
                    plt.plot(
                        [j + 0.5],
                        [i + 0.5],
                        color="orange",
                        marker="o",
                        markersize=10,
                    )

        plt.show()

    def draw_vertices(self, loops=None):
        m = self.problem.m
        n = self.problem.n_sites

        plt.xticks(range(n))
        plt.yticks(range(2 * m))
        # plt.grid()
        plt.xlim(0, n)
        plt.ylim(0, 2 * m)

        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("sites ($i$)")
        ax.set_ylabel("imaginary time ($j$)")

        # If there are loops to draw, draw them first
        if loops is not None and loops[0] is not None:
            for visited_vertex in loops:
                # Mark the start point
                i0, j0 = visited_vertex[0]
                plt.plot(
                    [j0],
                    [i0],
                    color="limegreen",
                    marker="o",
                    markersize=12,
                )

                # Draw the path
                i0, j0 = visited_vertex[-1]
                for i, j in visited_vertex:
                    if j0 == 0 and j == n - 1:
                        j0 = n
                    elif j0 == n - 1 and j == 0:
                        j = n

                    if i0 == 0 and i == 2 * m - 1:
                        i0 = 2 * m
                    elif i0 == 2 * m - 1 and i == 0:
                        i = 2 * m

                    i1, j1 = min(i0, i), min(j0, j)
                    ki, kj = 1, 1
                    if i == i0:
                        if i == 0:
                            i, i0, i1 = 2 * m, 2 * m, 2 * m
                        ki = -1 if (i + j1) % 2 == 1 else 1

                    if j == j0:
                        if j == 0:
                            j, j0, j1 = n, n, n
                        kj = -1 if (i1 + j) % 2 == 1 else 1

                    plt.plot(
                        [j0, j1 + kj * 0.5, j],
                        [i0, i1 + ki * 0.5, i],
                        color="green",
                        linewidth=3,
                    )
                    i0, j0 = i % (2 * m), j % n

        spins = self.spins

        tiles = np.zeros((2 * m, n))
        for j in range(2 * m):
            for i in range(n):
                tiles[j, i] = (
                    i + j % 2 + 1
                ) % 2  # checkerboard pattern for better visibility

        plt.imshow(
            tiles,
            cmap="gray",
            interpolation="nearest",
            vmin=-0.1,
            vmax=1,
            extent=(0, n, 2 * m, 0),
        )

        for j in range((2 * m) + 1):
            for i in range(n + 1):
                if spins[j % (2 * m), i % n] == 1:
                    plt.plot(i, j, marker="o", color="red", markersize=5)
                else:
                    plt.plot(i, j, marker="o", color="cornflowerblue", markersize=5)

        for j in range((2 * m) + 1):
            for base_i in range((n // 2) + 1):
                i = 2 * base_i + (j % 2)

                bot_left = np.array([j, i])
                bot_right = np.array([j, i + 1])
                top_right = np.array([j + 1, i + 1])
                top_left = np.array([j + 1, i])
                center = np.array([j + 0.5, i + 0.5])

                if spins[top_left[0] % (2 * m), top_left[1] % n] == -1:
                    direction = center - top_left
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        pivot="tip",
                        color="grey",
                    )
                else:
                    direction = top_left - center
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="black",
                    )

                if spins[top_right[0] % (2 * m), top_right[1] % n] == -1:
                    direction = center - top_right
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        pivot="tip",
                        color="grey",
                    )
                else:
                    direction = top_right - center
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="black",
                    )

                if spins[bot_right[0] % (2 * m), bot_right[1] % n] == -1:
                    direction = bot_right - center
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="grey",
                    )
                else:
                    direction = center - bot_right
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="black",
                        pivot="tip",
                    )

                if spins[bot_left[0] % (2 * m), bot_left[1] % n] == -1:
                    direction = bot_left - center
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="grey",
                    )
                else:
                    direction = center - bot_left
                    plt.quiver(
                        center[1],
                        center[0],
                        direction[1],
                        direction[0],
                        scale=1.5,
                        scale_units="xy",
                        color="black",
                        pivot="tip",
                    )

        plt.show()

    def _initialize_state(self):
        i, j0 = 0, 0
        set_spins = [0] * self.problem.n_sites
        spins = np.zeros((int(2 * self.problem.m), self.problem.n_sites), dtype=int)
        spins, _, _ = self.generate_valid_state(spins, i, j0, set_spins)
        return spins

    def generate_valid_state(self, spins, i, j0, set_spins):  # with prints
        m = int(2 * self.problem.m)
        n = self.problem.n_sites
        print(f"drawing at i={i}, j0={j0}")
        self.draw(spins)

        # If we're at the top row and the current cell is unset, choose a random spin value
        if i == 0:
            set_spins[j0] = 1

        if i == 0 and spins[i, j0] == 0:
            print(f"i=0, j0 = {j0} unset spin")
            set_spins[j0] = 1
            chosen_value = random.choice([-1, 1])
            print(f"i=0, j0 = {j0} unset spin, chose value ", chosen_value)
            new_spins = spins.copy()
            new_spins[i, j0] = chosen_value
            new_set_spins = set_spins.copy()
            new_spins, new_set_spins, validity = self.generate_valid_state(
                new_spins, i, j0, new_set_spins
            )
            if validity:
                print(f"i=0, j0 = {j0} chosen value {chosen_value} is valid")
                return new_spins, new_set_spins, True
            else:
                print(
                    f"i=0, j0 = {j0} chosen value {chosen_value} is not valid, new value is {-chosen_value}"
                )
                chosen_value = -chosen_value
                spins[0, j0] = chosen_value
                return self.generate_valid_state(spins, i, j0, set_spins)

        print(f"i= {i}, j0 = {j0}, testing neighbor")
        k = 1 - 2 * ((i + j0) % 2)  # k = 1 if same parity, -1 if different parity
        neighbor = (j0 + k) % n

        parity = (
            spins[i, j0] * spins[i, neighbor]
        )  # 1 if both have the same spin, -1 if opposite spins, 0 if neighbor is unset

        if i == m - 1:
            print(f"i= {i}, j0 = {j0}, at last row")
            if spins[(i + 1) % m, j0] != 0 and spins[(i + 1) % m, neighbor] != 0:
                print(f"i= {i}, j0 = {j0}, at last row, both below spins set")
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
                    print(
                        f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, looking for new j"
                    )
                    j = 0
                    while j < n and set_spins[j % n] == 1:
                        j += 1
                    if set_spins[j % n] == 0:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, found new j = {j}"
                        )
                        return self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins
                        )
                    else:
                        print("all spins are set, returning valid state")
                        return spins, set_spins, True

            if spins[(i + 1) % m, j0] == 0 and spins[(i + 1) % m, neighbor] == 0:
                print(f"i= {i}, j0 = {j0}, at last row, both below spins unset")
                chosen_spin = random.choice([j0, neighbor])
                new_spins = spins.copy()
                new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                new_set_spins = set_spins.copy()
                print(
                    f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin {chosen_spin}"
                )

                if new_set_spins[chosen_spin] == 0:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin"
                    )
                    new_set_spins[chosen_spin] = 1
                    new_spins, new_set_spins, validity = self.generate_valid_state(
                        new_spins, (i + 1) % m, chosen_spin, new_set_spins
                    )

                else:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j"
                    )
                    j = 0
                    while new_set_spins[j % n] == 1 and j < n:
                        j += 1
                    if new_set_spins[j % n] == 0:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}"
                        )
                        new_spins, new_set_spins, validity = self.generate_valid_state(
                            new_spins, (i + 1) % m, j, new_set_spins
                        )
                    else:
                        print("all spins are set, returning valid state")
                        return new_spins, new_set_spins, True

                if validity:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin {chosen_spin}, valid"
                    )
                    return new_spins, new_set_spins, True
                else:
                    chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                    print(
                        f"i= {i}, j0 = {j0}, at last row, both below spins unset, chose spin  not valid, chosen spin {chosen_spin} instead"
                    )
                    spins[(i + 1) % m, chosen_spin] = spins[i, j0]

            elif spins[(i + 1) % m, j0] != 0 or spins[(i + 1) % m, neighbor] != 0:
                print(f"i= {i}, j0 = {j0}, at last row, one below spin set")
                if parity == 1:
                    print(f"i= {i}, j0 = {j0}, at last row, parity=1")
                    if spins[(i + 1) % m, j0] == 0:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=1, j0 unset, setting spin at j0"
                        )
                        spins[(i + 1) % m, j0] = spins[i, j0]
                        chosen_spin = j0

                    else:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=1, j0 set, invalid state"
                        )
                        return spins, set_spins, False

                if parity == -1:
                    chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
                    print(
                        f"i= {i}, j0 = {j0}, at last row, parity=-1, setting spin at {chosen_spin} as its the one empty"
                    )
                    spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                if parity == 0:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, parity=0, checking if the set spin is compatible"
                    )
                    j1 = j0 if spins[(i + 1) % m, j0] != 0 else neighbor
                    if spins[(i + 1) % m, j1] != spins[i, j0]:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is opposite, its a valid state, setting the other spin"
                        )
                        chosen_spin = j0 if j1 != j0 else neighbor
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    elif spins[(i + 1) % m, j1] == spins[i, j0] and j1 == j0:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is same and below, only one option as we cannot cross with same spins"
                        )
                        chosen_spin = j0
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    else:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is same, need to try both options, going to the set spin or to the other as both are possible"
                        )
                        chosen_spin = random.choice([j0, neighbor])
                        new_spins = spins.copy()
                        new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                        new_set_spins = set_spins.copy()
                        print(
                            f"i= {i}, j0 = {j0}, at last row, parity=0, trying chosen spin {chosen_spin}"
                        )

                        if new_set_spins[chosen_spin] == 0:
                            print(
                                f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin"
                            )
                            new_set_spins[chosen_spin] = 1
                            new_spins, new_set_spins, validity = (
                                self.generate_valid_state(
                                    new_spins, (i + 1) % m, chosen_spin, new_set_spins
                                )
                            )

                        else:
                            print(
                                f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j"
                            )
                            j = 0
                            while new_set_spins[j % n] == 1 and j < n:
                                j += 1
                            if new_set_spins[j % n] == 0:
                                print(
                                    f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}"
                                )
                                new_spins, new_set_spins, validity = (
                                    self.generate_valid_state(
                                        new_spins, (i + 1) % m, j, new_set_spins
                                    )
                                )
                            else:
                                print("all spins are set, returning valid state")
                                return new_spins, new_set_spins, True

                        if validity:
                            print(
                                f"i= {i}, j0 = {j0}, at last row, parity=0, chosen spin {chosen_spin} is valid"
                            )
                            return new_spins, new_set_spins, True
                        else:
                            print(
                                f"i= {i}, j0 = {j0}, at last row, parity=0, chosen spin {chosen_spin} is not valid, trying the other spin"
                            )
                            chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                            spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                print(
                    f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} now seeing if it has already been set  before"
                )
                if set_spins[chosen_spin] == 0:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} hasn't been set, setting it now and continue on that spin"
                    )
                    set_spins[chosen_spin] = 1
                    return self.generate_valid_state(
                        spins, (i + 1) % m, chosen_spin, set_spins
                    )

                else:
                    print(
                        f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, looking for new j"
                    )
                    j = 0
                    while set_spins[j % n] == 1 and j < n:
                        j += 1
                    if set_spins[j % n] == 0:
                        print(
                            f"i= {i}, j0 = {j0}, at last row, chosen spin is {chosen_spin} has already been set, found new j = {j}"
                        )
                        return self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins
                        )
                    else:
                        print("all spins are set, returning valid state")
                        return spins, set_spins, True

        if parity == 0:
            print(f"i= {i}, j0 = {j0}, parity=0")
            chosen_spin = random.choice([j0, neighbor])
            new_spins = spins.copy()
            new_spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            new_set_spins = set_spins.copy()
            print(f"i= {i}, j0 = {j0}, parity=0, trying chosen spin {chosen_spin}")
            new_spins, new_set_spins, validity = self.generate_valid_state(
                new_spins, (i + 1) % m, chosen_spin, new_set_spins
            )
            if validity:
                print(
                    f"i= {i}, j0 = {j0}, parity=0, chosen spin {chosen_spin} is valid"
                )
                return new_spins, new_set_spins, True
            else:
                print(
                    f"i= {i}, j0 = {j0}, parity=0, chosen spin {chosen_spin} is not valid, trying the other spin"
                )
                chosen_spin = [neighbor, j0][1 - (chosen_spin == j0)]
                spins[(i + 1) % m, chosen_spin] = spins[i, j0]
                return self.generate_valid_state(
                    spins, (i + 1) % m, chosen_spin, set_spins
                )

        if parity == 1:
            print(f"i= {i}, j0 = {j0}, parity=1")
            if spins[(i + 1) % m, j0] == 0:
                print(f"i= {i}, j0 = {j0}, parity=1, j0 unset, setting spin at j0")
                spins[(i + 1) % m, j0] = spins[i, j0]
                return self.generate_valid_state(spins, (i + 1) % m, j0, set_spins)

            else:
                print(f"i= {i}, j0 = {j0}, parity=1, j0 set, invalid state")
                return spins, set_spins, False

        if parity == -1:
            print(f"i= {i}, j0 = {j0}, parity=-1, going to set the only available spin")
            chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
            spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            return self.generate_valid_state(spins, (i + 1) % m, chosen_spin, set_spins)

        print(" WARNING Should not reach here")


class ExhaustiveWorldline:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.worldlines = self.generate_all_wordlines()

    def generate_all_wordlines(self):
        # ensure we start with an empty result list on each call
        worldlines = []
        # set to record seen configurations (avoid duplicates)
        spins = np.zeros((int(2 * self.problem.m), self.problem.n_sites), dtype=int)
        set_spins = [0] * self.problem.n_sites
        i, j0 = 0, 0
        self.generate_valid_state(spins, i, j0, set_spins, worldlines)
        return worldlines

    def generate_valid_state(self, spins, i, j0, set_spins, worldlines):
        m = int(2 * self.problem.m)
        n = self.problem.n_sites
        # self.draw(spins)

        # If we're at the top row and the current cell is unset, choose a random spin value
        if i == 0:
            set_spins[j0] = 1
            # At top row: if unset, branch both spin choices; otherwise mark column as seen and continue
            if spins[i, j0] == 0:
                # First we test spin = 1
                new_spins1 = spins.copy()
                new_spins1[i, j0] = 1
                new_set_spins1 = set_spins.copy()
                self.generate_valid_state(new_spins1, i, j0, new_set_spins1, worldlines)

                # Then we test spin = -1
                new_spins2 = spins.copy()
                new_spins2[i, j0] = -1
                new_set_spins2 = set_spins.copy()
                self.generate_valid_state(new_spins2, i, j0, new_set_spins2, worldlines)
                return

        k = 1 - 2 * ((i + j0) % 2)  # k = 1 if same parity, -1 if different parity
        neighbor = (j0 + k) % n

        parity = (
            spins[i, j0] * spins[i, neighbor]
        )  # 1 if both have the same spin, -1 if opposite spins, 0 if neighbor is unset

        if i == m - 1:
            # print("i=m-1")
            if spins[(i + 1) % m, j0] != 0 and spins[(i + 1) % m, neighbor] != 0:
                if (
                    spins[(i + 1) % m, j0]
                    * spins[i, j0]
                    * spins[(i + 1) % m, neighbor]
                    * spins[(i + 1) % m, neighbor]
                    == 1
                ):
                    # print("i=m-1, on cherche un nouveau j")
                    j = 0
                    while j < n and set_spins[j % n] == 1:
                        j += 1

                    if set_spins[j % n] == 0:
                        self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins, worldlines
                        )
                    else:
                        worldlines.append(spins.copy())

                return

            if spins[(i + 1) % m, j0] == 0 and spins[(i + 1) % m, neighbor] == 0:
                # Try in the same column
                new_spins1 = spins.copy()
                new_spins1[(i + 1) % m, j0] = spins[i, j0]
                new_set_spins1 = set_spins.copy()

                # Check if we filled all columns or if its the first time we arrive at this column
                if new_set_spins1[j0] == 0:
                    new_set_spins1[j0] = 1
                    self.generate_valid_state(
                        new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines
                    )

                else:
                    j = 0
                    while j < n and new_set_spins1[j % n] == 1:
                        j += 1

                    if new_set_spins1[j % n] == 0:
                        self.generate_valid_state(
                            new_spins1, (i + 1) % m, j, new_set_spins1, worldlines
                        )

                    else:
                        worldlines.append(new_spins1.copy())

                # Try in the neighbor column
                new_spins2 = spins.copy()
                new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
                new_set_spins2 = set_spins.copy()

                # Check if we filled all columns or if its the first time we arrive at this column
                if new_set_spins2[neighbor] == 0:
                    new_set_spins2[neighbor] = 1
                    self.generate_valid_state(
                        new_spins2, (i + 1) % m, neighbor, new_set_spins2, worldlines
                    )

                else:
                    j = 0
                    while j < n and new_set_spins2[j % n] == 1:
                        j += 1

                    if new_set_spins2[j % n] == 0:
                        self.generate_valid_state(
                            new_spins2, (i + 1) % m, j, new_set_spins2, worldlines
                        )

                    else:
                        worldlines.append(spins.copy())
                return

            elif spins[(i + 1) % m, j0] != 0 or spins[(i + 1) % m, neighbor] != 0:
                if parity == 1:
                    # both columns have same spin: set the row below for both columns if needed
                    if spins[(i + 1) % m, j0] == 0:
                        spins[(i + 1) % m, j0] = spins[i, j0]
                        chosen_spin = j0

                    else:
                        return

                if parity == -1:
                    chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
                    spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                if parity == 0:
                    # one of the below row cells may already be set and possibly incompatible
                    j1 = j0 if spins[(i + 1) % m, j0] != 0 else neighbor
                    if spins[(i + 1) % m, j1] != spins[i, j0]:
                        chosen_spin = j0 if j1 != j0 else neighbor
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    elif spins[(i + 1) % m, j1] == spins[i, j0] and j1 == j0:
                        chosen_spin = j0
                        spins[(i + 1) % m, chosen_spin] = spins[i, j0]

                    else:
                        # First j0
                        new_spins1 = spins.copy()
                        new_spins1[(i + 1) % m, j0] = spins[i, j0]
                        new_set_spins1 = set_spins.copy()

                        # Check if we filled all columns or if its the first time we arrive at this column
                        if new_set_spins1[j0] == 0:
                            new_set_spins1[j0] = 1
                            self.generate_valid_state(
                                new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines
                            )

                        else:
                            j = 0
                            while j < n and new_set_spins1[j % n] == 1:
                                j += 1

                            if new_set_spins1[j % n] == 0:
                                self.generate_valid_state(
                                    new_spins1,
                                    (i + 1) % m,
                                    j,
                                    new_set_spins1,
                                    worldlines,
                                )

                            else:
                                worldlines.append(new_spins1.copy())

                        # Try in the neighbor column
                        new_spins2 = spins.copy()
                        new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
                        new_set_spins2 = set_spins.copy()

                        # Check if we filled all columns or if its the first time we arrive at this column
                        if new_set_spins2[neighbor] == 0:
                            new_set_spins2[neighbor] = 1
                            self.generate_valid_state(
                                new_spins2,
                                (i + 1) % m,
                                neighbor,
                                new_set_spins2,
                                worldlines,
                            )

                        else:
                            j = 0
                            while j < n and new_set_spins2[j % n] == 1:
                                j += 1

                            if new_set_spins2[j % n] == 0:
                                self.generate_valid_state(
                                    new_spins2,
                                    (i + 1) % m,
                                    j,
                                    new_set_spins2,
                                    worldlines,
                                )

                            else:
                                worldlines.append(spins.copy())
                        return

                if set_spins[chosen_spin] == 0:
                    set_spins[chosen_spin] = 1
                    self.generate_valid_state(
                        spins, (i + 1) % m, chosen_spin, set_spins, worldlines
                    )
                    return

                else:
                    j = 0
                    while j < n and set_spins[j] == 1:
                        j += 1
                    if set_spins[j % n] == 0:
                        self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins, worldlines
                        )

                    else:
                        worldlines.append(spins.copy())
                    return

        if parity == 0:
            # try j0
            new_spins1 = spins.copy()
            new_spins1[(i + 1) % m, j0] = spins[i, j0]
            new_set_spins1 = set_spins.copy()
            self.generate_valid_state(
                new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines
            )

            new_spins2 = spins.copy()
            new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
            new_set_spins2 = set_spins.copy()
            self.generate_valid_state(
                new_spins2, (i + 1) % m, neighbor, new_set_spins2, worldlines
            )
            return

        if parity == 1:
            # straight continuation: set downward cell and recurse if the two cells do not cross
            if spins[(i + 1) % m, j0] == 0:
                spins[(i + 1) % m, j0] = spins[i, j0]
                self.generate_valid_state(spins, (i + 1) % m, j0, set_spins, worldlines)
            return

        if parity == -1:
            # both continuations are possible, choose the only available cell
            chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
            spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            self.generate_valid_state(
                spins, (i + 1) % m, chosen_spin, set_spins, worldlines
            )
            return
        print(" WARNING Should not reach here")

    def draw_worldline(self, grid):
        """
        Draw a n x m grid of random black (-1) and white (1) squares.
        """
        n = self.problem.n_sites
        m = int(2 * self.problem.m)
        spin_color = {1: "red", -1: "cornflowerblue", 0: "black"}

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
