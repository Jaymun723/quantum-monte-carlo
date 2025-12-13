"""
# Computations of Energy, Spins with perform_vertex_loop_update, fixed T

Short name: `local_n_J_x_J_z_T` (names of fixed parameters)
"""

from production import MonteCarlo, Problem
from production.local_updates import local_shift_move
from pathlib import Path

ms = [2, 4, 6, 8, 10, 12]
problems = [Problem(n_sites=8, J_x=2.2, J_z=1.0, temperature=1.0, m=m) for m in ms]

mc = MonteCarlo(
    problems,
    local_shift_move,
    n_cycles=5_000,
    n_rep=10,
    save_folder=Path("./data/local_n_J_x_J_z_T"),
)

mc.run()
