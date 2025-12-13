from production.loop_updates import loop_update
from production import Problem, ExactSolver, Worldline
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

ms = [2, 4, 6, 8, 10, 12]
problems = [Problem(n_sites=8, J_x=30, J_z=1, temperature=10, m=m) for m in ms]
rng = np.random.default_rng(7)


solver = ExactSolver(problems[0])

n_rep = 10
n_cycles = 5_000


energies = np.zeros((len(problems), n_rep, n_cycles))
times = np.zeros((len(problems), n_rep))
for problem_idx, pb in enumerate(problems):
    length_cycle = 2 * pb.m * pb.n_sites
    for r in range(n_rep):
        t0 = time()
        spins = np.ones((2 * pb.m, pb.n_sites))
        for i in range(0, pb.n_sites, 2):
            spins[:, i] *= -1

        wl = Worldline(pb, spins)

        for _ in range(1_000):
            loop_update(wl, rng)

        with tqdm(
            total=n_cycles * length_cycle,
            unit="step",
            desc=f"Monte Carlo m={pb.m} (rep {r})",
        ) as pbar:
            for i in range(n_cycles):
                for _ in range(length_cycle):
                    loop_update(wl, rng)
                    pbar.update(1)
                energies[problem_idx, r, i] = wl.compute_energy()
        times[problem_idx, r] = time() - t0

fig, ax = plt.subplots()
ax.set_title("Time of computation in function of m")
ax.boxplot(times.T, positions=ms)
ax.plot(ms, np.mean(times, axis=1), linestyle="dashed")
ax.set_ylabel("Time of computation (s)")
ax.set_xlabel("m")
np.save("times.npy", times)
fig.savefig("times.png")
plt.show()

fig, ax = plt.subplots()
ax.set_title("Energy computations")
ax.hlines(
    solver.energy.real,
    min(ms),
    max(ms),
    color="green",
    label="Exact energy",
)
ax.plot(ms, (np.mean(energies, axis=(1, 2))), linestyle="dashed")
ax.boxplot(np.mean(energies, axis=2).T, positions=ms, label="Monte Carlo Energy")
ax.set_xlabel("m")
ax.set_ylabel("Energy")
ax.legend(loc="lower right")
np.save("energies.npy", energies)
fig.savefig("energies.png")
plt.show()