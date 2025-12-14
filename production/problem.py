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
        Calculates the loop update probabilities for the XXZ model.
        """
        W1 = self.weight_side
        W2 = -self.weight_cross
        W3 = self.weight_full

        if (W1 > W2 + W3) or (W2 > W1 + W3) or (W3 > W1 + W2):
            raise ValueError("Incorrect weight configurarion for loop updates.")

        d = 0.5 * (W2 + W3 - W1)
        v = 0.5 * (W1 + W3 - W2)
        h = 0.5 * (W1 + W2 - W3)

        probs_S1 = {"G1": v / W1, "G2": h / W1, "G3": 0.0}
        probs_S1 = (
            list(probs_S1.keys()),
            list(probs_S1.values()),
        )

        probs_S2 = {"G2": h / W2, "G4": d / W2, "G3": 0.0}
        probs_S2 = (
            list(probs_S2.keys()),
            list(probs_S2.values()),
        )

        probs_S3 = {"G1": v / W3, "G4": d / W3, "G3": 0.0}
        probs_S3 = (
            list(probs_S3.keys()),
            list(probs_S3.values()),
        )

        return {
            "S1": probs_S1,
            "S2": probs_S2,
            "S3": probs_S3,
        }

    def __repr__(self):
        return f"Problem(n={self.n_sites}, J_x={self.J_x}, J_z={self.J_z}, T={self.temperature}, m={self.m})"
