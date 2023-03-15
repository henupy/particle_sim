"""
File for the Ball-class used in the simulation
"""

from __future__ import annotations

import numpy as np
import common.misc as misc

# Shortcut for typing
numeric = int | float


class Ball:
    def __init__(self, p0: np.ndarray, v0: np. ndarray, r: numeric) -> None:
        """
        :param p0: Initial position of the ball [m]
        :param v0: Initial velocity of the ball [m/s]
        :param r: Radius of the ball [m]
        """
        self.pos = p0
        self.v = v0
        self.r = r

    def _apply_boundaries(self, w: numeric, h: numeric) -> None:
        """
        Collides the ball with the boundaries
        :param w: Width of the box [m]
        :param h: Height of the box [m]
        :return:
        """
        x, y = self.pos
        # Bottom x-boundary
        if (x - self.r) < 0:
            self.v[0] = -self.v[0]
            self.pos[0] = self.r
        # Top x-boundary
        if (x + self.r) > w:
            self.v[0] = -self.v[0]
            self.pos[0] = w - self.r
        # Bottom y-boundary
        if (y - self.r) < 0:
            self.v[1] = -self.v[1]
            self.pos[1] = self.r
        # Top y-boundary
        if (y + self.r) > h:
            self.v[1] = -self.v[1]
            self.pos[1] = y - self.r

    def _end_velocity(self, other: Ball) -> None:
        """
        Calculates the end velocity for both balls after their collision
        :param other: The ball that our ball has collided
        :return:
        """
        # Shorthands
        v1, v2 = self.v, other.v
        v12 = v1 - v2
        v21 = v2 - v1
        r12 = self.pos - other.pos
        r21 = other.pos - self.pos
        # End velocities for both balls
        self.v = v1 - (np.dot(v12, r12) / misc.sq_len(r12)) * r12
        other.v = v2 - (np.dot(v21, r21) / misc.sq_len(r12)) * r21

    def _apply_collisions(self, others: list[Ball]) -> None:
        """
        Applies collisions between the balls. As of right now, only takes into
        account a single collision, i.e., a collision with only one other ball.
        :param others: List of all the balls the ball under inspection could
        collide
        :return:
        """
        for ball in others:
            dist = misc.vec_len(ball.pos - self.pos)
            rsum = self.r + ball.r
            if dist <= rsum:
                self._end_velocity(ball)
                return

    def update_position(self, others: list, dt: numeric, w: numeric, h: numeric) -> None:
        """
        :param others: List of the other balls
        :param dt: Timestep [m]
        :param w: Width of the box [m]
        :param h: Height of the box [m]
        :return:
        """
        self._apply_collisions(others=others)
        self._apply_boundaries(w=w, h=h)
        self.pos += self.v * dt

    def __eq__(self, other: Ball) -> bool:
        """
        Two balls are equal if they have the same position, radius, and velocity
        :param other:
        :return:
        """
        return all(self.pos == other.pos) and all(self.v == other.v) \
            and self.r == other.r
