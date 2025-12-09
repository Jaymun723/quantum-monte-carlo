from production.problem import Problem
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MonteCarlo:
    def __init__(
        self,
        problem: Problem,
        update_fn,
        n_cycles: int,
        n_rep: int,
        seed: int = 7,
        save_folder: Path = Path("."),
    ):
        self.pb = problem
        self.length_cycle = 2 * problem.m * problem.n_sites
        self.update_fn = update_fn
        self.n_cycles = n_cycles
        self.n_rep = n_rep
        self.rng = np.random.default_rng(seed)
        self.save_folder = save_folder
