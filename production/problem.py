import numpy as np
import functools


class Problem:
    """
    The problem class holds all the parameters for a simulation.
    """

    def __init__(
        self,
        n_sites: int,
        J_x: float,
        J_z: float,
        temperature: float,
        m: int,
        k_b: float = 1.0,
    ):
        self.n_sites = n_sites
        self.J_x = J_x
        self.J_z = J_z
        self.temperature = temperature
        self.m = m
        self.k_b = k_b

    @functools.cached_property
    def delta_tau(self):
        return self.beta / self.m

    @functools.cached_property
    def beta(self):
        return 1 / (self.k_b * self.temperature)

    @functools.cached_property
    def weight_full(self):
        return np.exp(-self.delta_tau * self.J_z / 4.0)

    @functools.cached_property
    def energy_full(self):
        return self.J_z / 4

    @functools.cached_property
    def weight_cross(self):
        return -np.exp(self.delta_tau * self.J_z / 4.0) * np.sinh(
            self.delta_tau * self.J_x / 2.0
        )

    @functools.cached_property
    def energy_cross(self):
        return -(self.J_z / 4) - (self.J_x / 2) / (
            np.tanh(self.delta_tau * self.J_x / 2)
        )

    @functools.cached_property
    def weight_side(self):
        return np.exp(self.delta_tau * self.J_z / 4.0) * np.cosh(
            self.delta_tau * self.J_x / 2.0
        )

    @functools.cached_property
    def energy_side(self):
        return -(self.J_z / 4) - (self.J_x / 2) * np.tanh(self.delta_tau * self.J_x / 2)

    @functools.cached_property
    def loop_probabilities(self):
        """
        Calculates the generalized loop update probabilities for the XXZ model,
        allowing for the "frozen" graph (G=3) to handle all parameter regimes.
        """
        # 1. Calculate Vertex Weights W(S)
        W1 = self.weight_side
        W2 = -self.weight_cross
        W3 = self.weight_full

        # 2. Determine the Frozen Graph Weight (f)
        # We set f to ensure all other weights (v, h, d) remain non-negative.
        # The condition is effectively checking the "triangle inequality" of weights.

        # Calculate potential negative deficits
        deficit_diag = W1 - (W2 + W3)
        deficit_vert = W2 - (W1 + W3)
        deficit_horiz = W3 - (W1 + W2)

        # Set f to cover the largest deficit (if any)
        f = max(0, deficit_diag, deficit_vert, deficit_horiz)

        # 3. Calculate Graph Weights using the general solution
        d = 0.5 * (W2 + W3 - W1 + f)  # Diagonal
        v = 0.5 * (W1 + W3 - W2 + f)  # Vertical
        h = 0.5 * (W1 + W2 - W3 + f)  # Horizontal

        # 4. Calculate Probabilities P(S -> (S, G))

        # Vertex S=1 (Néel): Compatible with Vertical, Horizontal, Frozen
        probs_S1 = {"G1": v / W1, "G2": h / W1, "G3": f / W1}

        # Vertex S=2 (Flip): Compatible with Horizontal, Diagonal, Frozen
        # Guard against W2=0 (pure Ising case)
        if W2 > 1e-14:
            probs_S2 = {"G2": h / W2, "G4": d / W2, "G3": f / W2}
        else:
            probs_S2 = {"G2": 0, "G4": 0, "G3": 0}

        # Vertex S=3 (Ferro): Compatible with Vertical, Diagonal, Frozen
        probs_S3 = {"G1": v / W3, "G4": d / W3, "G3": f / W3}

        return {
            "S1": probs_S1,
            "S2": probs_S2,
            "S3": probs_S3,
        }

    def __repr__(self):
        return f"Problem(n={self.n_sites}, J_x={self.J_x}, J_z={self.J_z}, T={self.temperature}, m={self.m})"