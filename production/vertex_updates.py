from .worldline import Worldline
import numpy as np
from functools import reduce

def random_white_plaquette(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    i, j = rng.integers(0, 2 * m), rng.integers(0, n//2)
    return i%(2*m), (2*j + i%2)%n, 

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


    if w.spins[i0, j0] == 1:
        directions.append((i0, j0, up, left))
    
    if w.spins[i0, j1] == 1:
        directions.append((i0, j1, up, right))
    
    if w.spins[i1, j0] == -1:
        directions.append((i1, j0, down, left))
    
    if w.spins[i1, j1] == -1:
        directions.append((i1, j1, down, right))

    return directions
     
def vertex_move(w: Worldline, rng: np.random.Generator, visited_vertex: list, visited_plaquettes: set, i, j):
    m = w.problem.m  # i
    n = w.problem.n_sites  # j
       

    if (i, j) in visited_plaquettes:
        return (i, j), 0, 1, True
    
    directions = available_directions_in_plaquette(w, i, j)

    # choose a random direction to move to
    i0, j0, i_new, j_new = directions[rng.integers(0, len(directions))]
    visited_vertex.append((i0, j0))
    visited_plaquettes.add((i, j))

    #calculate the new energy and weight changes
    dE, dOmega = 0.0, 1.0
    if len(visited_vertex) > 0:
        plaquette = np.array([[w.spins[i, j], w.spins[i, (j + 1) %n]],
                              [w.spins[(i + 1) % (2 * m), j], w.spins[(i + 1) % (2 *m), (j + 1) % n]]])
        plaquette_new = plaquette.copy()
        plaquette_new[(i0-i) % 2, (j0-j) % 2] *= -1
        i_old, j_old = visited_vertex[-1]
        plaquette_new[(i_old - i)%2, (j_old - j)%2] *= -1
        dE, dOmega = parameters_for_plaquette_configuration(w, plaquette )
        dE_new, dOmega_new = parameters_for_plaquette_configuration(w, plaquette_new)
        dE = dE_new - dE
        dOmega = dOmega_new / dOmega



    

    return (i_new, j_new), dE, dOmega, False

def vertex_loop(w: Worldline, rng: np.random.Generator):
    n = w.problem.n_sites  # j
    m = w.problem.m  # i

    visited_vertex = []
    visited_plaquettes = set()
    n_vertices = 0
    max_vertices = n * (2*m)  #safety limit to avoid infinite loops

    i, j = random_white_plaquette(w, rng)
    already_visited = False
    delta_E = 0.0
    delta_Omega = 1.0

    while not already_visited and n_vertices < max_vertices:
        (i, j), dE, dOmega, already_visited = vertex_move(w, rng, visited_vertex, visited_plaquettes,  i, j)
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
    n = w.problem.n_sites
    m = w.problem.m

    visited_vertex, delta_E, delta_Omega = vertex_loop(w, rng)

    dE = 0.0
    dOmega = 1.0

    #adding the first plaquette to the energy and weight changes
    i0, j0 = visited_vertex[0]
    i1, j1 = visited_vertex[-1]
    i, j = min(i0, i1), min(j0, j1)
    plaquette = np.array([[w.spins[i, j], w.spins[i, (j + 1) %n]], [w.spins[(i + 1) % (2 * m), j], w.spins[(i + 1) % (2 * m), (j + 1) % n]]])
    plaquette_new = plaquette.copy()
    plaquette_new[(i0 - i)%2, (j0 - j)%2] *= -1
    plaquette_new[(i1 - i)%2, (j1 - j)%2] *= -1
    dE, dOmega = parameters_for_plaquette_configuration(w, plaquette)
    dE_new, dOmega_new = parameters_for_plaquette_configuration(w, plaquette_new)
    dE = dE_new - dE
    dOmega = dOmega_new / dOmega

    delta_E += dE
    delta_Omega *= dOmega
    

    if True: #rng.random() < np.abs(delta_Omega):
        w.energy += delta_E
        w.weight *= delta_Omega
        for i, j in visited_vertex:
            w.spins[i, j] *= -1
        

    
   
