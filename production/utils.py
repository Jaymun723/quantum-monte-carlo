import numpy as np


def draw_key(probs: tuple[list[str], list[float]], rng: np.random.Generator) -> str:
    """
    Selects a key based on the probability distribution.
    """
    a = rng.random()
    if a < probs[1][0]:
        return probs[0][0]
    elif a < probs[1][0] + probs[1][1]:
        return probs[0][1]
    else:
        return probs[0][2]


class GridUnionFind:
    """Simple UnionFind structure adapted for the grid of spins."""

    def __init__(self, m, n):
        self.rows = 2 * m
        self.cols = n

        size = self.rows * self.cols
        self.parent = list(range(size))
        self.rank = [0] * size

    def _get_index(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Coordinate ({r}, {c}) out of bounds.")
        return r * self.cols + c

    def find(self, r, c):
        i = self._get_index(r, c)

        if self.parent[i] != i:
            # Path Compression
            self.parent[i] = self._find_by_index(self.parent[i])
        return self.parent[i]

    def _find_by_index(self, i):
        if self.parent[i] != i:
            self.parent[i] = self._find_by_index(self.parent[i])
        return self.parent[i]

    def union(self, r1, c1, r2, c2):
        i = self._get_index(r1, c1)
        j = self._get_index(r2, c2)

        root_i = self._find_by_index(i)
        root_j = self._find_by_index(j)

        if root_i != root_j:
            # Union by Rank
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

    def get_ensembles(self):
        """
        Returns the different partitions / ensembles.
        """
        from collections import defaultdict

        groups = defaultdict(list)

        for r in range(self.rows):
            for c in range(self.cols):
                root_id = self.find(r, c)
                groups[root_id].append((r, c))

        return list(groups.values())


def get_mean_err(energies: np.ndarray, reverse: bool = False):
    mean_energies = np.mean(energies, axis=(1, 2))
    err_energies = np.std(np.mean(energies, axis=2), axis=1) / np.sqrt(
        energies.shape[1]
    )
    if reverse:
        return mean_energies[::-1], err_energies[::-1]
    return mean_energies, err_energies
