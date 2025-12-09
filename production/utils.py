import numpy as np


def draw_key(
    probability_dict: dict[str, float], rng: np.random.Generator = None
) -> str:
    """
    Selects a key based on the probability distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    keys = list(probability_dict.keys())
    probs = list(probability_dict.values())

    return rng.choice(keys, p=probs)


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
