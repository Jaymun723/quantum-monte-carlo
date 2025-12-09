"""
# Computations of Energy, Spins with loop_updates, fixed T

Short name: `loop_n_J_z_T_m` (names of fixed parameters)
"""

from production import MonteCarlo, Problem, loop_update
from pathlib import Path
import numpy as np

J_xs = np.linspace(2.0, 10.0, 10)
problems = [Problem(n_sites=8, J_x=J_x, J_z=1.0, temperature=1.0, m=6) for J_x in J_xs]

mc = MonteCarlo(
    problems,
    loop_update,
    n_cycles=5_000,
    n_rep=10,
    save_folder=Path("../data/loop_n_J_x_J_z_T"),
)

mc.run()
