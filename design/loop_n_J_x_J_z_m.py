"""
# Computations of Energy, Spins with loop_updates, fixed m

Short name: `loop_n_J_x_J_z_m` (names of fixed parameters)
"""

from production import MonteCarlo, Problem, loop_update
from pathlib import Path
import numpy as np

temperatures = np.concat([np.linspace(0.1, 10, 7), np.linspace(20, 100, 3)])
problems = [
    Problem(n_sites=8, J_x=2.2, J_z=1.0, temperature=t, m=6) for t in temperatures
]

mc = MonteCarlo(
    problems,
    loop_update,
    n_cycles=5_000,
    n_rep=10,
    save_folder=Path("./data/loop_n_J_x_J_z_m"),
)

mc.run()
