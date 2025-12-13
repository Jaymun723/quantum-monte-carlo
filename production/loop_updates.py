from production.utils import draw_key, GridUnionFind
from production import Worldline
import numpy as np


def get_loops(wl: Worldline, rng: np.random.Generator):
    """
    Maps graphs to the Worldline then returns the loops.
    """
    n = wl.problem.n_sites
    m = wl.problem.m

    spins = wl.spins

    uf = GridUnionFind(m, n)

    probs = wl.problem.loop_probabilities

    for j in range(2 * m):
        for base_i in range(n // 2):
            i = 2 * base_i + (j % 2)

            j_plus = (j + 1) % (2 * m)
            i_plus = (i + 1) % n

            if spins[j, i] != spins[j_plus, i]:  # cross, S2
                key = draw_key(probs["S2"], rng)
                if key == "G2":
                    uf.union(j, i, j, i_plus)
                    uf.union(j_plus, i, j_plus, i_plus)
                elif key == "G4":
                    uf.union(j, i, j_plus, i_plus)
                    uf.union(j, i_plus, j_plus, i)
                else:
                    print("aïe G3")
            elif spins[j, i] != spins[j, i_plus]:  # side, S1
                key = draw_key(probs["S1"], rng)
                if key == "G1":
                    uf.union(j, i, j_plus, i)
                    uf.union(j, i_plus, j_plus, i_plus)
                elif key == "G2":
                    uf.union(j, i, j, i_plus)
                    uf.union(j_plus, i, j_plus, i_plus)
                else:
                    print("aïe G3")
            else:  # full, S3
                key = draw_key(probs["S3"], rng)
                if key == "G1":
                    uf.union(j, i, j_plus, i)
                    uf.union(j, i_plus, j_plus, i_plus)
                elif key == "G4":
                    uf.union(j, i, j_plus, i_plus)
                    uf.union(j, i_plus, j_plus, i)
                else:
                    print("aïe G3")

    return uf.get_ensembles()


def loop_update(wl: Worldline, rng: np.random.Generator):
    """
    Performs the algorithm described in section 1.3.
    """
    loops = get_loops(wl, rng)

    for loop in loops:
        if rng.random() < 0.5:
            for j, i in loop:
                wl.spins[j, i] *= -1
