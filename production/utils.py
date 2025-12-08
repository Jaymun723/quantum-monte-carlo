import numpy as np


def draw_key(probability_dict: dict[str, float], rng: np.random.Generator = None):
    """
    Selects a key based on the probability distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    keys = list(probability_dict.keys())
    probs = list(probability_dict.values())

    return rng.choice(keys, p=probs)


class GridUnionFind:
    def __init__(self, m, n):
        # Grid dimensions: height is 2*m, width is n
        self.rows = 2 * m
        self.cols = n

        size = self.rows * self.cols
        self.parent = list(range(size))
        self.rank = [0] * size

    def _get_index(self, r, c):
        """Helper to convert 2D (row, col) to 1D index."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Coordinate ({r}, {c}) out of bounds.")
        return r * self.cols + c

    def find(self, r, c):
        """Finds the representative of the cell (r, c)."""
        i = self._get_index(r, c)

        if self.parent[i] != i:
            # Path Compression
            self.parent[i] = self._find_by_index(self.parent[i])
        return self.parent[i]

    def _find_by_index(self, i):
        """Internal find helper for path compression recursion."""
        if self.parent[i] != i:
            self.parent[i] = self._find_by_index(self.parent[i])
        return self.parent[i]

    def union(self, r1, c1, r2, c2):
        """Unions the sets containing cells (r1, c1) and (r2, c2)."""
        # Convert coordinates to indices
        i = self._get_index(r1, c1)
        j = self._get_index(r2, c2)

        # Find roots using internal index logic
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
        Returns a list of lists, where each inner list contains
        the (r, c) coordinates of a connected component.
        """
        from collections import defaultdict

        # Dictionary to group cells by their root ID
        groups = defaultdict(list)

        for r in range(self.rows):
            for c in range(self.cols):
                # find(r, c) returns the unique ID of the set this cell belongs to
                root_id = self.find(r, c)
                groups[root_id].append((r, c))

        # Return just the lists of coordinates (the values of the dict)
        return list(groups.values())
