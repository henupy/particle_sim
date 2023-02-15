"""
File containg the Particle-object used in the Brownian
motion simulation
"""

from __future__ import annotations

import numpy as np

# Shortcuts for typing
numeric = int | float


def _vec_len(v: np.ndarray) -> numeric:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.sum(np.power(v, 2)))


class Particle:
    k = 1.390658e-23  # Boltzmann's constant [J/K]

    def __init__(self, pos0: np.ndarray, v0: np. ndarray, mass: numeric) -> None:
        """
        Initialize the Particle-object
        :param pos0: Initial position of the particle
        :param v0: Initial velocity
        :param mass: Mass of the particle
        :param mass: Mass of the particle in kilograms
        """
        self.pos = pos0
        self.v = v0
        self.mass = mass

    def _vel2temp(self) -> numeric:
        """
        Converts the particle's velocity to temperature
        :return:
        """
        ke = 0.5 * self.mass * _vec_len(self.v)
        return ke / self.k

    def _apply_boundaries(self, w: numeric, h: numeric) -> None:
        """
        Collides the particle with the boundaries
        :param w: Width of the box
        :param h: Height of the box
        :return:
        """
        if not (0 < self.pos[0] < w):
            self.v[0] = -self.v[0]
        if not (0 < self.pos[1] < h):
            self.v[1] = -self.v[1]

    def _end_velocity(self, other: Particle, direc: np.ndarray) -> None:
        """
        Calculates the end velocity for both particles after their collision
        :param other: The particle that our particle has collided
        :param direc: The vector connecting the particles' centers of mass
        :return:
        """
        # Shorthands
        v1, v2 = self.v, other.v
        v12 = v1 - v2
        v21 = v2 - v1
        m1, m2 = self.mass, other.mass
        # Solution
        v1 = v1 + 2 * (np.dot(direc, v12)) / (m1 * (1 / m1 + 1 / m2))
        v2 = v2 + 2 * (np.dot(direc, v21)) / (m2 * (1 / m1 + 1 / m2))
        self.v = v1
        other.v = v2

    def _apply_collisions(self, others: list, epsilon: numeric) -> None:
        """
        Applies collisions between particles
        :param others: List of all the particles the particle under inspection can
        collide
        :param epsilon: If the distance between particles is below this value,
        the particles are deemed to collide. Essentially the sum of the particles'
        radiuses.
        :return:
        """
        for p in others:
            direc = p.pos - self.pos
            dist = _vec_len(direc)
            if dist < epsilon:
                self._end_velocity(p, direc)
                return

    def update_position(self, others: list, dt: numeric, epsilon: numeric,
                        w: numeric, h: numeric) -> None:
        """
        :param others: List of the other particles
        :param dt: Timestep
        :param epsilon:
        :param w: Width of the box
        :param h: Height of the box
        :return:
        """
        self._apply_boundaries(w, h)
        self._apply_collisions(others=others, epsilon=epsilon)
        self.pos += self.v * dt
