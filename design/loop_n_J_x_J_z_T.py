"""
# Computations of Energy, Spins with loop_updates, fixed T

Short name: `loop_n_J_x_J_z_T` (names of fixed parameters)
"""

from production import MonteCarlo, Problem, loop_update
from pathlib import Path

ms = [6]
problems = [Problem(n_sites=8, J_x=2.2, J_z=1.0, temperature=1.0, m=m) for m in ms]

mc = MonteCarlo(
    problems,
    loop_update,
    n_cycles=5_000,
    n_rep=1,
    save_folder=Path("./data/loop_n_J_x_J_z_T"),
)

mc.run()
