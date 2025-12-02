from . import Wordline, Problem
import numpy as np
import itertools


def random_wordline(p: Problem, rng: np.random.Generator | None = None):
    # mettre l'algo de théa à terme, mais pour l'instant celui là fonctionne
    rng = np.random.default_rng() if rng is None else rng
    total_configs = 2 * p.m * p.n_sites
    flats_configs = list(itertools.product([1, -1], repeat=total_configs))
    rng.shuffle(flats_configs)

    for flat_config in flats_configs:
        config = np.array(flat_config).reshape((2 * p.m, p.n_sites))
        w = Wordline(p, config)
        if w.weight != 0:
            return w

    raise ValueError("No configuration found.")


def local_move(w: Wordline, rng: np.random.Generator | None = None, switch_propa=0.2):
    rng = np.random.default_rng() if rng is None else rng

    n = w.problem.n_sites
    m = w.problem.m

    # Shift or switch ?
    if rng.random() < switch_propa:
        starting_cells = list(range(n))
        line = [(0, rng.choice(starting_cells))]
        while line[-1][0] != 0 or line[0][1] != line[-1][1] or len(line) == 1:
            # print(
            #     f"len(line) % (2 * m)={len(line) % (2 * m)}, line[0][1]={line[0][1]}, line[-1][1]={line[-1][1]}"
            # )
            j, i = line[-1]
            j_plus = (j + 1) % (2 * m)
            i_plus = (i + 1) % n
            i_minus = (n + i - 1) % n
            # print(f"j={j}, i={i}, j_plus={j_plus}, i_plus={i_plus}, i_minus={i_minus}")
            if w.spins[j, i] == w.spins[j_plus, i]:
                line.append((j_plus, i))
            elif i % 2 == j % 2:  # look at i+1
                assert w.spins[j, i] == w.spins[j_plus, i_plus]
                line.append((j_plus, i_plus))
            else:
                assert w.spins[j, i] == w.spins[j_plus, i_minus]
                line.append((j_plus, i_minus))
        return line[:-1]
