from . import Wordline, Problem
import numpy as np
import itertools
from functools import reduce


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


def local_line_move(w: Wordline, rng: np.random.Generator):
    n = w.problem.n_sites
    m = w.problem.m

    starting_cells = list(range(n))
    line = [(0, rng.choice(starting_cells), 0.0, 1.0)]
    # dE = 0.0
    # dOmega = 1.0
    while line[-1][0] != 0 or line[0][1] != line[-1][1] or len(line) == 1:
        j, i, _, _ = line[-1]
        j_plus = (j + 1) % (2 * m)
        i_plus = (i + 1) % n
        i_minus = (n + i - 1) % n
        # print(f"At j={j}, i={i}", end=" > ")
        if w.spins[j, i] == w.spins[j_plus, i]:
            dE = 0.0
            dOmega = 1.0

            i_next = i_minus  # on the left side of the white cell
            if j % 2 == i % 2:  # on the right side of the white cell
                i_next = i_plus

            if w.spins[j, i] == w.spins[j, i_next]:
                # print(
                #     f"from full to side {'(right)' if i_next == i_plus else '(left)'}"
                # )
                dE -= w.problem.energy_full
                dE += w.problem.energy_side
                dOmega /= w.problem.weight_full
                dOmega *= w.problem.weight_side
                # print(f"dE = {dE}, dOmega = {dOmega}")
            else:
                # print(
                #     f"from side {'(right)' if i_next == i_plus else '(left)'} to full"
                # )
                dE -= w.problem.energy_side
                dE += w.problem.energy_full
                dOmega /= w.problem.weight_side
                dOmega *= w.problem.weight_full
                # print(f"dE = {dE}, dOmega = {dOmega}")

            line.append((j_plus, i, dE, dOmega))
        elif i % 2 == j % 2:  # look at i+1
            assert w.spins[j, i] == w.spins[j_plus, i_plus]
            # we are on the right side of the white cell
            dE = 0.0
            dOmega = 1.0

            # print("from cross (right) to full")
            dE -= w.problem.energy_cross
            dE += w.problem.energy_full
            dOmega /= w.problem.weight_cross
            dOmega *= w.problem.weight_full
            # print(f"dE = {dE}, dOmega = {dOmega}")

            line.append((j_plus, i_plus, dE, dOmega))
        else:
            assert w.spins[j, i] == w.spins[j_plus, i_minus]
            dE = 0.0
            dOmega = 1.0

            # print("from cross (left) to full")
            dE -= w.problem.energy_cross
            dE += w.problem.energy_full
            dOmega /= w.problem.weight_cross
            dOmega *= w.problem.weight_full
            # print(f"dE = {dE}, dOmega = {dOmega}")
            line.append((j_plus, i_minus, dE, dOmega))

    delta_E = sum([dE for _, _, dE, _ in line]) / w.problem.m

    if rng.random() < np.exp(-w.problem.beta * delta_E):
        # prev_energy = w.energy
        # prev_weight = w.weight
        # new_w = w.copy()
        delta_Omega = reduce(
            lambda x, y: y[3] * x, line, 1.0
        )  # product of all the dOmega

        for j, i, _, _ in line[:-1]:
            w.spins[j, i] *= -1
        # new_energy = w.compute_energy()
        # new_weight = w.compute_weight()
        # print(f"{new_energy - prev_energy} == {delta_E}")
        # print(f"{new_weight / prev_weight} == {delta_Omega} == {delta_Omega}")
        # assert (new_energy - prev_energy) == delta_E
        # assert (new_weight / prev_weight) == delta_Omega
        w.energy += delta_E
        w.weight *= delta_Omega

    return w


def random_shaded_plaquette(w: Wordline, rng: np.random.Generator):
    m = w.problem.m
    n = w.problem.n_sites

    possibles_plaquettes = []

    for j in range(2 * m):
        for base_i in range(n // 2):
            i = 2 * base_i + (1 - (j % 2))
            j_plus = (j + 1) % (2 * m)
            i_plus = (i + 1) % n

            # print(f"Looking at plaquette {(j, i)}")

            if (
                w.spins[j, i] == w.spins[j_plus, i]
                and w.spins[j, i_plus] == w.spins[j_plus, i_plus]
                and w.spins[j, i] != w.spins[j, i_plus]
            ):
                # print("its good")
                possibles_plaquettes.append(
                    [(j, i), (j_plus, i), (j, i_plus), (j_plus, i_plus)]
                )

    if possibles_plaquettes == []:
        return None
    else:
        print(possibles_plaquettes)
        return possibles_plaquettes[rng.integers(len(possibles_plaquettes))]


def local_shift_move(w: Wordline, rng: np.random.Generator):
    cells = random_shaded_plaquette(w, rng)
    if cells is None:
        return w

    m = w.problem.m
    n = w.problem.n_sites

    j, i = cells[0]
    j_plus, i_plus = cells[3]
    j_two = (j_plus + 1) % (2 * m)
    j_minus = (2 * m + j - 1) % (2 * m)
    i_minus = (n + i - 1) % n
    i_two = (i_plus + 1) % n

    delta_E = 0.0
    delta_Omega = 1.0
    # Vertical shift
    if w.spins[j, i] == w.spins[j_minus, i]:
        delta_E += w.problem.energy_cross - w.problem.energy_side
        delta_Omega = delta_Omega * w.problem.weight_cross / w.problem.weight_side
    else:
        delta_E += w.problem.energy_side - w.problem.energy_cross
        delta_Omega = delta_Omega * w.problem.weight_side / w.problem.weight_cross
    if w.spins[j_plus, i] == w.spins[j_two, i]:
        delta_E += w.problem.energy_cross - w.problem.energy_side
        delta_Omega = delta_Omega * w.problem.weight_cross / w.problem.weight_side
    else:
        delta_E += w.problem.energy_side - w.problem.energy_cross
        delta_Omega = delta_Omega * w.problem.weight_side / w.problem.weight_cross
    # Horizontal shift
    if w.spins[j, i] == w.spins[j, i_minus]:
        delta_E += w.problem.energy_side - w.problem.energy_full
        delta_Omega = delta_Omega * w.problem.weight_side / w.problem.weight_full
    else:
        delta_E += w.problem.energy_full - w.problem.energy_side
        delta_Omega = delta_Omega * w.problem.weight_full / w.problem.weight_side
    if w.spins[j, i_plus] == w.spins[j, i_two]:
        delta_E += w.problem.energy_side - w.problem.energy_full
        delta_Omega = delta_Omega * w.problem.weight_side / w.problem.weight_full
    else:
        delta_E += w.problem.energy_full - w.problem.energy_side
        delta_Omega = delta_Omega * w.problem.weight_full / w.problem.weight_side
    delta_E /= m

    # prev_energy = w.energy
    # prev_weight = w.weight
    if rng.random() < np.exp(-w.problem.beta * delta_E):
        w.energy += delta_E
        w.weight *= delta_Omega
        for cell in cells:
            w.spins[cell] *= -1
    # new_energy = w.compute_energy()
    # new_weight = w.compute_weight()

    # print(new_energy - prev_energy, delta_E)
    # print(new_weight / prev_weight, delta_Omega)

    return w


def local_move(w: Wordline, rng: np.random.Generator | None = None, switch_propa=1.0):
    rng = np.random.default_rng() if rng is None else rng

    # Shift or switch ?
    if rng.random() < switch_propa:
        return local_line_move(w, rng)
    else:
        return local_shift_move(w, rng)
