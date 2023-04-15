"""
File for the Path-class used in the simulation
"""

import numpy as np


class Path:
    def __init__(self, v0: np.ndarray, dt: int | float,
                 v1: np.ndarray = None, t_coll: int | float = None) -> None:
        """
        :param v0: Initial velocity of the object (at time t0) [m/s]
        :param dt: The length of the total timestep [s]
        :param v1: The new velocity of the object after the collision [m/s]
        :param t_coll: The time of the collision (0 ... dt) [s]
        :return:
        """
        self.v0 = v0
        self.dt = dt
        self.v1 = v1
        self.t_coll = t_coll

    def execute(self) -> np.ndarray:
        """
        :return:
        """
        if self.v1 is None or self.t_coll is None:
            return self.v0 * self.dt
        pos = self.v0 * self.t_coll
        pos += self.v1 * (self.dt - self.t_coll)
        return pos
