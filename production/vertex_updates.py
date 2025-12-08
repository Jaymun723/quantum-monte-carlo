from .worldline import Worldline
import numpy as np
from functools import reduce

def random_white_plaquette(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    i = int(rng.integers(0, 2 * m))
    j = int(rng.integers(0, max(1, n//2)))
    return i % (2 * m), (2 * j + i % 2) % n

def available_directions_in_plaquette(w: Worldline, i: int, j: int):
    """
    Given a plaquette (i, j), return the list of available plaquettes to move to.
    """
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    directions = []
    i0, j0 = i%(2*m), j % n # left and up coordinates
    i1, j1 = (i + 1) % (2 * m), (j + 1) % n # right and down coordinates
    down, right = (i + 1) % (2 * m), (j + 1) % n # right and down plaquettes
    up, left = (i - 1) % (2 * m), (j - 1) % n # left and up plaquettes


    if w.spins[i0, j0] == -1:
        directions.append((i0, j0, up, left))
    
    if w.spins[i0, j1] == -1:
        directions.append((i0, j1, up, right))
    
    if w.spins[i1, j0] == 1:
        directions.append((i1, j0, down, left))
    
    if w.spins[i1, j1] == 1:
        directions.append((i1, j1, down, right))

    return directions
     
def vertex_move(w: Worldline, rng: np.random.Generator, visited_vertex: list, seen_vertex: set, initial_plaquette, i, j):
    m = w.problem.m  # i
    n = w.problem.n_sites  # j
       
    # If we are back to the initial plaquette we stop
    if (i, j) == initial_plaquette and len(visited_vertex) > 0:
        i0, j0 = visited_vertex[0]
        i_old, j_old = visited_vertex[-1]
        plaquette = np.array([[w.spins[i, j], w.spins[i, (j + 1) % n]],
                              [w.spins[(i + 1) % (2 * m), j], w.spins[(i + 1) % (2 * m), (j + 1) % n]]])
        plaquette_new = plaquette.copy()
        # flip the first vertex within the plaquette
        plaquette_new[(i0 - i) % 2, (j0 - j) % 2] *= -1
        # flip the previously visited vertex if present (the last element)
        plaquette_new[(i_old - i) % 2, (j_old - j) % 2] *= -1
        dE, dOmega = parameters_for_plaquette_configuration(w, plaquette)
        dE_new, dOmega_new = parameters_for_plaquette_configuration(w, plaquette_new)
        dE = dE_new - dE
        dOmega = dOmega_new / dOmega
        return (i, j), dE, dOmega, True
    
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
        # print(visited_vertex, directions[idx][2:])
        directions = d
        idx = int(rng.integers(0, len(directions)))
    i0, j0, i_new, j_new = directions[idx]
    visited_vertex.append((i0, j0))
    seen_vertex.add((i0, j0))

    #calculate the new energy and weight changes
    dE, dOmega = 0.0, 1.0
    if len(visited_vertex) > 0:
        # compute the plaquette at the current (i, j)
        plaquette = np.array([[w.spins[i, j], w.spins[i, (j + 1) % n]],
                              [w.spins[(i + 1) % (2 * m), j], w.spins[(i + 1) % (2 * m), (j + 1) % n]]])
        plaquette_new = plaquette.copy()
        # flip the chosen vertex within the plaquette
        plaquette_new[(i0 - i) % 2, (j0 - j) % 2] *= -1
        # flip the previously visited vertex if present (the element before the last)
        if len(visited_vertex) >= 2:
            i_old, j_old = visited_vertex[-2]
            plaquette_new[(i_old - i) % 2, (j_old - j) % 2] *= -1
        dE, dOmega = parameters_for_plaquette_configuration(w, plaquette)
        dE_new, dOmega_new = parameters_for_plaquette_configuration(w, plaquette_new)
        dE = dE_new - dE
        dOmega = dOmega_new / dOmega



    

    return (i_new, j_new), dE, dOmega, False

def vertex_loop(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    visited_vertex = []
    seen_vertex = set()
    initial_plaquette = random_white_plaquette(w, rng)
    n_vertices = 0
    max_vertices = n * (2*m)  #safety limit to avoid infinite loops

    i, j = initial_plaquette
    already_visited = False
    delta_E = 0.0
    delta_Omega = 1.0

    while not already_visited and n_vertices < max_vertices:
        (i, j), dE, dOmega, already_visited = vertex_move(w, rng, visited_vertex, seen_vertex, initial_plaquette,  i, j)
        delta_E += dE
        delta_Omega *= dOmega
        n_vertices += 1
    
    if n_vertices == max_vertices:
        print("Warning: maximum number of vertices reached in vertex loop update.")
        return None, None, None
    
    return visited_vertex, delta_E, delta_Omega

def parameters_for_plaquette_configuration(w: Worldline, plaquette):
    dE, dOmega = 0.0, 1.0

    #parities of the spins in the plaquette
    a = plaquette[0, 0]*plaquette[0, 1]
    # b = plaquette[1, 0]*plaquette[1, 1]
    d = plaquette[0, 1]*plaquette[1, 1]
    c = plaquette[0, 0]*plaquette[1, 0]

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

    visited_vertex, delta_E, delta_Omega = vertex_loop(w, rng)

    # if vertex loop failed or returned nothing
    if visited_vertex is None:
        return None

    if len(visited_vertex) == 0:
        return None

    
    

    if True: #rng.random() < np.abs(delta_Omega):
        w.energy += delta_E
        w.weight *= delta_Omega
        for i, j in visited_vertex:
            w.spins[i, j] *= -1
        return visited_vertex
    
    return None
        

    
   
