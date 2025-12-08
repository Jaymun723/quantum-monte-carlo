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
        return np.exp(-self.delta_tau * self.J_z / 4)

    @functools.cached_property
    def energy_full(self):
        return self.J_z / 4

    @functools.cached_property
    def weight_cross(self):
        return -np.exp(self.delta_tau * self.J_z / 4) * np.sinh(
            self.delta_tau * self.J_x / 2
        )

    @functools.cached_property
    def energy_cross(self):
        return -(self.J_z / 4) - (self.J_x / 2) / (
            np.tanh(self.delta_tau * self.J_x / 2)
        )

    @functools.cached_property
    def weight_side(self):
        return np.exp(self.delta_tau * self.J_z / 4) * np.cosh(
            self.delta_tau * self.J_x / 2
        )

    @functools.cached_property
    def energy_side(self):
        return -(self.J_z / 4) - (self.J_x / 2) * np.tanh(self.delta_tau * self.J_x / 2)
