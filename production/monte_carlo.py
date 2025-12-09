from production.problem import Problem
from production.worldline import Worldline
import numpy as np
from tqdm import tqdm
from pathlib import Path
from time import time
import pickle


class MonteCarlo:
    def __init__(
        self,
        problems: list[Problem],
        update_fn,
        n_cycles: int,
        n_rep: int,
        seed: int = 7,
        save_folder: Path = Path("."),
    ):
        self.problems = problems
        self.update_fn = update_fn
        self.n_cycles = n_cycles
        self.n_rep = n_rep
        self.rng = np.random.default_rng(seed)
        self.save_folder = save_folder
        
        self.energies = np.zeros((len(problems), n_rep, n_cycles))
        self.spins = []
        for pb_idx, pb in enumerate(self.problems):
            self.spins.append([])
            for r in range(n_rep):
                self.spins[pb_idx].append([])
                for _ in range(self.n_cycles):
                    self.spins[pb_idx][r].append([])
        
        self.times = np.zeros((len(problems), n_rep))

    def run_problem(self, pb_idx: int):
        pb = self.problems[pb_idx]
        length_cycle = 2 * pb.m * pb.n_sites
        for r in range(self.n_rep):
            t0 = time()
            spins = np.ones((2 * pb.m, pb.n_sites))
            for i in range(0, pb.n_sites, 2):
                spins[:, i] *= -1

            wl = Worldline(pb, spins)

            for _ in range(1_000):
                self.update_fn(wl, self.rng)

            with tqdm(
                total=self.n_cycles * length_cycle,
                unit="step",
                desc=f"{pb} ({r} of {self.n_rep})",
            ) as pbar:
                for i in range(self.n_cycles):
                    for _ in range(length_cycle):
                        self.update_fn(wl, self.rng)
                        pbar.update(1)
                    self.energies[pb_idx, r, i] = wl.compute_energy()

                    for d in range(1, pb.n_sites):
                        spins_corr = np.zeros((2*pb.m, pb.n_sites))
                        for j in range(2*pb.m):
                            for s in range(pb.n_sites):
                                spins_corr[j, s] = wl.spins[j, s] * wl.spins[j, (s + d) % pb.n_sites]
                        self.spins[pb_idx][r][i].append(np.mean(spins_corr))

            self.times[pb_idx, r] = time() - t0
        
        np.save(self.save_folder / "energies.npy", self.energies)
        with open( self.save_folder / "spins.pickle") as f:
            pickle.dump(self.spins, f)
        np.save(self.save_folder / "times.npy", self.times)

    def run(self):
        for i in range(len(self.problems)):
            self.run_problem(i)

