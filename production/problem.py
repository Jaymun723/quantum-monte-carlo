class Problem:
    """
    The problem class holds all the parameters for a simulation.
    """
    def __init__(self, n_sites: int, J_x: float, J_z: float, temperature: float, m: int, k_b: float = 1.0):
        self.n_sites = n_sites
        self.J_x = J_x
        self.J_z = J_z
        self.temperature = temperature
        self.m = m
        self.k_b = k_b

    @property
    def delta_tau(self):
        return self.beta / self.m

    @property
    def beta(self):
        return 1 / (self.k_b * self.temperature)