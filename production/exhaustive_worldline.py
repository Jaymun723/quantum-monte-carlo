from .problem import Problem
import numpy as np
import matplotlib.pyplot as plt


class ExhaustiveWorldline:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.worldlines = self.generate_all_wordlines()

    def generate_all_wordlines(self):
        # ensure we start with an empty result list on each call
        worldlines = []
        # set to record seen configurations (avoid duplicates)
        w_set = set()
        spins = np.zeros((int(2 * self.problem.m), self.problem.n_sites), dtype=int)
        set_spins = [0] * self.problem.n_sites
        i, j0 = 0, 0
        self.generate_valid_state(spins, i, j0, set_spins, worldlines, w_set)
        return worldlines

    def generate_valid_state(self, spins, i, j0, set_spins, worldlines, w_set):
        m = int(2 * self.problem.m)
        n = self.problem.n_sites
        # print(f"drawing at i={i}, j0={j0}")
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
                new_set_spins1[j0] = 1
                self.generate_valid_state(new_spins1, 0, j0, new_set_spins1, worldlines, w_set)
                # print(f"i=0, j0 = {j0} unset spin, choosen 1 finished")

                # Then we test spin = -1 
                # print(f"i=0, j0 = {j0} unset spin, choose -1")
                new_spins2 = spins.copy()
                new_spins2[i, j0] = -1
                new_set_spins2 = set_spins.copy()
                new_set_spins2[j0] = 1
                self.generate_valid_state(new_spins2, 0, j0, new_set_spins2, worldlines, w_set)
                # print(f"i=0, j0 = {j0} unset spin, choosen -1 finished")
                return

        # print(f"i= {i}, j0 = {j0}, testing neighbor")
        k = 1 - 2 * ((i + j0) % 2)  # k = 1 if same parity, -1 if different parity
        neighbor = (j0 + k) % n

        parity = (
            spins[i, j0] * spins[i, neighbor]
        )  # 1 if both have the same spin, -1 if opposite spins, 0 if neighbor is unset
        # print("parity =", parity)

        if i == m - 1:
            # print(f"i=m-1, j0 = {j0}, at last row ")
            if spins[(i + 1) % m, j0] != 0 and spins[(i + 1) % m, neighbor] != 0:
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins set")
                if (
                    spins[(i + 1) % m, j0]
                    * spins[i, j0]
                    * spins[(i + 1) % m, neighbor]
                    * spins[(i + 1) % m, neighbor]
                    == -1
                ):
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, incompatible spins")
                    True 
            
                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, looking for new j")
                    j = 0
                    while j < n and set_spins[j % n] == 1:
                        j += 1

                    if set_spins[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, both below spins set, compatible spins, found new j = {j}")
                        self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins, worldlines, w_set
                        )
                        return
                    else:
                        # print("all spins are set, returning valid state")
                        # self.draw(spins)
                        key = spins.tobytes()
                        if key not in w_set:
                            worldlines.append(spins.copy())
                            w_set.add(key)
                        return

            if spins[(i + 1) % m, j0] == 0 and spins[(i + 1) % m, neighbor] == 0:
                # Try in the same column 
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset, trying in j0 = {j0}")

                new_spins1 = spins.copy()
                new_spins1[(i + 1) % m, j0] = spins[i, j0]
                new_set_spins1 = set_spins.copy()

                # Check if we filled all columns or if its the first time we arrive at this column
                if new_set_spins1[j0] == 0:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {j0} hasn't been set, setting it now and continue on that spin")
                    new_set_spins1[j0] = 1
                    self.generate_valid_state(
                    new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines, w_set
                    )
                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {j0} has already been set, looking for new j")
                    j = 0
                    while j < n and new_set_spins1[j % n] == 1:
                        j += 1

                    if new_set_spins1[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {j0} has already been set, found new j = {j}")
                        self.generate_valid_state(
                            new_spins1, (i + 1) % m, j, new_set_spins1, worldlines, w_set
                        )

                    else:
                        # print("all spins are set, returning valid state")
                        key = new_spins1.tobytes()
                        if key not in w_set:
                            worldlines.append(new_spins1)
                            w_set.add(key)
                        return


                # Try in the neighbor column
                # print(f"i= {i}, j0 = {j0}, at last row, both below spins unset, trying in neighbour = {neighbor}")
                new_spins2 = spins.copy()
                new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
                new_set_spins2 = set_spins.copy()

                # Check if we filled all columns or if its the first time we arrive at this column
                if new_set_spins2[neighbor] == 0:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {neighbor} hasn't been set, setting it now and continue on that spin")
                    new_set_spins2[neighbor] = 1
                    self.generate_valid_state(
                    new_spins2, (i + 1) % m, neighbor, new_set_spins2, worldlines, w_set
                    )
                else:
                    # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {neighbor} has already been set, looking for new j")
                    j = 0
                    while j < n and new_set_spins2[j % n] == 1:
                        j += 1

                    if new_set_spins2[j % n] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is {neighbor} has already been set, found new j = {j}")
                        self.generate_valid_state(
                            new_spins2, (i + 1) % m, j, new_set_spins2, worldlines, w_set
                        )

                    else:
                        # print("all spins are set, returning valid state")
                        # self.draw(new_spins2)
                        key = new_spins2.tobytes()
                        if key not in w_set:
                            worldlines.append(new_spins2)
                            w_set.add(key)
                        return
                return

            elif spins[(i + 1) % m, j0] != 0 or spins[(i + 1) % m, neighbor] != 0:
                # print(f"i= {i}, j0 = {j0}, at last row, one below spin set")
                if parity == 1:
                    # print(f"i= {i}, j0 = {j0}, at last row, parity=1")
                    # both columns have same spin: set the row below for both columns if needed
                    if spins[(i + 1) % m, j0] == 0:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=1, j0 unset, setting spin at j0")
                        spins[(i + 1) % m, j0] = spins[i, j0]
                        chosen_spin = j0

                    else:
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=1, j0 set, invalid state")
                        return

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
                        # First j0
                        # print("first try j0")
                        new_spins1 = spins.copy()
                        new_spins1[(i + 1) % m, j0] = spins[i, j0]
                        new_set_spins1 = set_spins.copy()

                        # Check if we filled all columns or if its the first time we arrive at this column
                        if new_set_spins1[j0] == 0:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is j0 = {j0} hasn't been set, setting it now and continue on that spin")
                            new_set_spins1[j0] = 1
                            self.generate_valid_state(
                            new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines, w_set
                            )

                        else:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is j0 = {j0} has already been set, looking for new j")
                            j = 0
                            while j < n and new_set_spins1[j % n] == 1:
                                j += 1

                            if new_set_spins1[j % n] == 0:
                                # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is j0 = {j0} has already been set, found new j = {j}")
                                self.generate_valid_state(
                                    new_spins1, (i + 1) % m, j, new_set_spins1, worldlines, w_set
                                )

                            else:
                                # print("all spins are set, returning valid state")
                                # self.draw(new_spins1)
                                worldlines.append(new_spins1)
                                return

                        # Try in the neighbor column
                        # print(f"i= {i}, j0 = {j0}, at last row, parity=0, set spin is same, need to try both options, going to the set spin or to the other as both are possible, now trying neighbor = {neighbor}")
                        new_spins2 = spins.copy()
                        new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
                        new_set_spins2 = set_spins.copy()

                        # Check if we filled all columns or if its the first time we arrive at this column
                        if new_set_spins2[neighbor] == 0:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is neighbor is {neighbor} hasn't been set, setting it now and continue on that spin")
                            new_set_spins2[neighbor] = 1
                            self.generate_valid_state(
                            new_spins2, (i + 1) % m, neighbor, new_set_spins2, worldlines, w_set
                            )

                        else:
                            # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is neighbor is {neighbor} has already been set, looking for new j")
                            j = 0
                            while j < n and new_set_spins2[j % n] == 1:
                                j += 1

                            if new_set_spins2[j % n] == 0:
                                # print(f"i= {i}, j0 = {j0}, at last row, chosen spin is neighbor is {neighbor} has already been set, found new j = {j}")
                                self.generate_valid_state(
                                    new_spins2, (i + 1) % m, j, new_set_spins2, worldlines, w_set
                                )
                                

                            else:
                                # print("all spins are set, returning valid state")
                                # self.draw(new_spins2)
                                worldlines.append(new_spins2)
                                return

                        return
                        

                if set_spins[chosen_spin] == 0:
                    set_spins[chosen_spin] = 1
                    self.generate_valid_state(
                        spins, (i + 1) % m, chosen_spin, set_spins, worldlines, w_set
                    )
                    return
                else:
                    j = 0
                    while j < n and set_spins[j] == 1:
                        j += 1
                    if set_spins[j % n] == 0:
                        self.generate_valid_state(
                            spins, (i + 1) % m, j, set_spins, worldlines, w_set
                        )
                        return
                    else:
                        key = spins.tobytes()
                        if key not in w_set:
                            worldlines.append(spins.copy())
                            w_set.add(key)
                        return
            return 
                    

        if parity == 0:
            # try j0
            new_spins1 = spins.copy()
            new_spins1[(i + 1) % m, j0] = spins[i, j0]
            new_set_spins1 = set_spins.copy()
            self.generate_valid_state(
                new_spins1, (i + 1) % m, j0, new_set_spins1, worldlines, w_set
            )

            # Try neighbor 
            new_spins2 = spins.copy()
            new_spins2[(i + 1) % m, neighbor] = spins[i, j0]
            new_set_spins2 = set_spins.copy()
            self.generate_valid_state(
                new_spins2, (i + 1) % m, neighbor, new_set_spins2, worldlines, w_set
            )
            return

        if parity == 1:
            # straight continuation: set downward cell and recurse if the two cells do not cross
            if spins[(i + 1) % m, j0] == 0:
                spins[(i + 1) % m, j0] = spins[i, j0]
                self.generate_valid_state(spins, (i + 1) % m, j0, set_spins, worldlines, w_set)
            return

        if parity == -1:
            # both continuations are possible, choose the only available cell
            chosen_spin = j0 if spins[(i + 1) % m, j0] == 0 else neighbor
            spins[(i + 1) % m, chosen_spin] = spins[i, j0]
            self.generate_valid_state(spins, (i + 1) % m, chosen_spin, set_spins, worldlines, w_set)
            return

    def draw(self, spins):
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
                plt.plot(j, i,marker='o', color = spin_color[spins[i % (2 * m), j % n]] , markersize=5)
                
                

        for i in range((2 * m)):
            for j in range(1, n):
                k = 1 - 2 * ((i%(2*m) + j) % 2)  # k = 1 if same parity, -1 if different parity
                neighbor = (j + k) % n
                i1, j1 = (i + 1) % (2 * m +1), (j + k) % (n+1)
                up = (i + 1) % (2 * m)

                c = spins[i%(2*m), j%n]*spins[up, j%n]
                d = spins[i%(2*m), neighbor]*spins[up, neighbor]
                

                
                if  c == 1 and d == 1:
                    # straight lines
                    plt.plot(
                        [j, j],
                        [i , i1],
                        color=spin_color[spins[i%(2*m), j%n]],
                        linewidth=2,
                    )
                    plt.plot(
                        [j1, j1],
                        [i, i1 ],
                        color=spin_color[spins[i%(2*m), neighbor]],
                        linewidth=2,
                    )

                if c == -1 and d == -1:
                    # crossed lines
                    plt.plot(
                        [j, j1],
                        [i , i1 ],
                        color=spin_color[spins[i%(2*m), j%n]],
                        linewidth=2,
                    )
                    plt.plot(
                        [j1, j],
                        [i , i1 ],
                        color=spin_color[spins[i%(2*m), neighbor]],
                        linewidth=2,
                    )

                if c*d == -1:
                    # Impossible square
                    plt.plot(
                        [j+0.5],
                        [i+0.5 ],
                        color='orange',
                        marker = 'o',
                        markersize=10,
                    )
                    

        

        plt.show()
