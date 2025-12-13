"""
# Computations of Energy, Spins with perform_vertex_loop_update, fixed T

Short name: `loop_J_x_J_z_T_m` (names of fixed parameters)
"""

from production import MonteCarlo, Problem, loop_update
from pathlib import Path

ns = [4, 6, 8, 10, 12, 14]
problems = [Problem(n_sites=n, J_x=2.2, J_z=1.0, temperature=1.0, m=6) for n in ns]

mc = MonteCarlo(
    problems,
    loop_update,
    n_cycles=5_000,
    n_rep=10,
    save_folder=Path("./data/loop_J_x_J_z_T_m_bis"),
)

mc.run()
