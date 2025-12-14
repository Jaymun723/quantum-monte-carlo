from .worldline import Worldline
import numpy as np


def random_white_plaquette(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    i = int(rng.integers(0, 2 * m))
    j = int(rng.integers(0, max(1, n // 2)))
    return i % (2 * m), (2 * j + i % 2) % n


def available_directions_in_plaquette(w: Worldline, i: int, j: int):
    """
    Given a plaquette (i, j), return the list of available plaquettes to move to.
    """
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    directions = []
    i0, j0 = i % (2 * m), j % n  # left and up coordinates
    i1, j1 = (i + 1) % (2 * m), (j + 1) % n  # right and down coordinates
    down, right = (i + 1) % (2 * m), (j + 1) % n  # right and down plaquettes
    up, left = (i - 1) % (2 * m), (j - 1) % n  # left and up plaquettes

    if w.spins[i0, j0] == -1:
        directions.append((i0, j0, up, left))

    if w.spins[i0, j1] == -1:
        directions.append((i0, j1, up, right))

    if w.spins[i1, j0] == 1:
        directions.append((i1, j0, down, left))

    if w.spins[i1, j1] == 1:
        directions.append((i1, j1, down, right))

    return directions


def vertex_move(
    w: Worldline,
    rng: np.random.Generator,
    visited_vertex: list,
    seen_vertex: set,
    visited_plaquettes,
    i,
    j,
):
    # m = w.problem.m  # i
    # n = w.problem.n_sites  # j

    # If we are back to the initial plaquette we stop
    if len(visited_vertex) > 1 and (i, j) == visited_plaquettes[0]:
        return (i, j), True

    visited_plaquettes.append((i, j))

    directions = available_directions_in_plaquette(w, i, j)

    # if no available directions, stop the loop
    if not directions:
        return (i, j), 0.0, 1.0, True

    # choose a random direction to move to

    idx = int(rng.integers(0, len(directions)))

    while directions[idx][:2] in seen_vertex:
        d = []
        for i in range(len(directions)):
            if i != idx:
                d.append(directions[i])
        directions = d
        idx = int(rng.integers(0, len(directions)))
    i0, j0, i_new, j_new = directions[idx]
    visited_vertex.append((i0, j0))
    seen_vertex.add((i0, j0))

    return (i_new, j_new), False


def vertex_loop(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    visited_vertex = []
    visited_plaquettes = []
    seen_vertex = set()
    initial_plaquette = random_white_plaquette(w, rng)
    n_vertices = 0
    max_vertices = n * (2 * m)  # safety limit to avoid infinite loops

    i, j = initial_plaquette
    already_visited = False

    while not already_visited and n_vertices < max_vertices:
        (i, j), already_visited = vertex_move(
            w, rng, visited_vertex, seen_vertex, visited_plaquettes, i, j
        )
        n_vertices += 1

    if n_vertices == max_vertices:
        print("Warning: maximum number of vertices reached in vertex loop update.")
        return None, None, None

    return visited_vertex, visited_plaquettes


def parameters_for_plaquette_configuration(w: Worldline, plaquette):
    dE, dOmega = 0.0, 1.0

    # parities of the spins in the plaquette
    a = plaquette[0, 0] * plaquette[0, 1]
    # b = plaquette[1, 0]*plaquette[1, 1]
    d = plaquette[0, 1] * plaquette[1, 1]
    c = plaquette[0, 0] * plaquette[1, 0]

    if a == 1 and c == 1 and d == 1:
        dE = w.problem.energy_full
        dOmega = w.problem.weight_full

    if a == -1 and c == 1 and d == 1:
        dE = w.problem.energy_side
        dOmega = w.problem.weight_side

    if a == -1 and c == -1 and d == -1:
        dE = w.problem.energy_cross
        dOmega = w.problem.weight_cross

    return dE, dOmega


def perform_vertex_loop_update(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i
    visited_vertex, visited_plaquettes = vertex_loop(w, rng)

    # if vertex loop failed or returned nothing
    if visited_vertex is None:
        return None

    if len(visited_vertex) == 0:
        return None

    spins2 = w.spins.copy()

    for i, j in visited_vertex:
        spins2[i, j] *= -1

    spins_copy = w.spins.copy()
    delta_E = 0
    delta_Omega = 1

    for i, j in visited_vertex:
        spins_copy[i, j] *= -1

    for i, j in visited_plaquettes:
        plaquette = np.array(
            [
                [w.spins[i, j], w.spins[i, (j + 1) % n]],
                [
                    w.spins[(i + 1) % (2 * m), j],
                    w.spins[(i + 1) % (2 * m), (j + 1) % n],
                ],
            ]
        )
        dE, dOmega = parameters_for_plaquette_configuration(w, plaquette)

        plaquette_copy = np.array(
            [
                [spins_copy[i, j], spins_copy[i, (j + 1) % n]],
                [
                    spins_copy[(i + 1) % (2 * m), j],
                    spins_copy[(i + 1) % (2 * m), (j + 1) % n],
                ],
            ]
        )
        dE_new, dOmega_new = parameters_for_plaquette_configuration(w, plaquette_copy)

        delta_E += (dE_new - dE) / m
        delta_Omega *= dOmega_new / dOmega

    if rng.random() < np.abs(delta_Omega):
        w.energy += delta_E
        w.weight *= delta_Omega
        w.spins = spins_copy
        return visited_vertex

    return None
