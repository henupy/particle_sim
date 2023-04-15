"""
File for the Ball-class used in the simulation
"""

from __future__ import annotations

import numpy as np
import common.misc as misc

from common.path import Path

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
        self.path = None

    def _apply_boundaries(self, w: numeric, h: numeric, dt: numeric) -> None:
        """
        Collides the ball with the boundaries. The collisions are handled so
        that if the ball is deemed to be out of bounds, a step is taken back in time,
        i.e., the ball is returned to its position in the previous timestep. After that,
        the collision time is calculated, and the ball is moved to the collision
        position. After that, the corresponding velocity component is flipped
        accordingly, and time is advanced to the end of the timestep
        :param w: Width of the box [m]
        :param h: Height of the box [m]
        :param dt: Timestep [s]
        :return: Whether a collision occurred or not
        """
        x, y = self.pos
        # Right boundary
        if (x - self.r) < 0:
            # Move the ball back to its location before the collision
            self.pos += -self.v * dt
            # Collision time
            dt_c = (self.r - self.pos[0]) / self.v[0]
            # New velocity after the collision
            v0 = self.v
            self.v = np.array([-self.v[0], self.v[1]])
            self.path = Path(v0=v0, dt=dt, v1=self.v, t_coll=dt_c)
            return
        # Left boundary
        if (x + self.r) > w:
            self.pos += -self.v * dt
            dt_c = (w - self.r - self.pos[0]) / self.v[0]
            v0 = self.v
            self.v = np.array([-self.v[0], self.v[1]])
            self.path = Path(v0=v0, dt=dt, v1=self.v, t_coll=dt_c)
            return
        # Bottom boundary
        if (y - self.r) < 0:
            self.pos += -self.v * dt
            dt_c = (self.r - self.pos[1]) / self.v[1]
            v0 = self.v
            self.v = np.array([self.v[0], -self.v[1]])
            self.path = Path(v0=v0, dt=dt, v1=self.v, t_coll=dt_c)
            return
        # Top boundary
        if (y + self.r) > h:
            self.pos += -self.v * dt
            dt_c = (h - self.r - self.pos[1]) / self.v[1]
            v0 = self.v
            self.v = np.array([self.v[0], -self.v[1]])
            self.path = Path(v0=v0, dt=dt, v1=self.v, t_coll=dt_c)
            return
        # If the ball did not collide with any boundaries, we can omit
        # the 'v1' and 't_coll' parameters from its Path object
        self.path = Path(v0=self.v, dt=dt)

    def _calc_velocity(self, other: Ball, dt: numeric) -> None:
        """
        Calculates the end velocity for both balls after their collision
        :param other: The ball that our ball has collided
        :param dt: Timestep [s]
        :return:
        """
        # TODO: 'Untangle' the balls after their collision and
        # TODO: and build their Paths correctly
        # Shorthands
        v1, v2 = self.v, other.v
        v12 = v1 - v2
        v21 = v2 - v1
        r12 = self.pos - other.pos
        r21 = other.pos - self.pos
        # The new velocity for our ball
        self.v = v1 - (np.dot(v12, r12) / misc.sq_len(r12)) * r12
        self.path = Path(v0=self.v, dt=dt)
        # The new velocity for the other ball
        other.v = v2 - (np.dot(v21, r21) / misc.sq_len(r21)) * r21
        other.path = Path(v0=other.v, dt=dt)

    def _apply_collisions(self, others: list[Ball], dt: numeric) -> None:
        """
        Applies collisions between the balls. As of right now, only takes into
        account a single collision, i.e., a collision with only one other ball.
        :param others: List of all the balls the ball under inspection could
        collide
        :param dt: Timestep [s]
        :return:
        """
        for ball in others:
            dist = misc.vec_len(ball.pos - self.pos)
            rsum = self.r + ball.r
            if dist <= rsum:
                self._calc_velocity(other=ball, dt=dt)
                # We assume that the ball collides with only one other ball
                return

    def create_path(self, others: list, dt: numeric, w: numeric,
                    h: numeric) -> None:
        """
        :param others:
        :param dt:
        :param w:
        :param h:
        :return:
        """
        self._apply_collisions(others=others, dt=dt)
        self._apply_boundaries(w=w, h=h, dt=dt)

    def __eq__(self, other: Ball) -> bool:
        """
        Two balls are equal if they have the same position, radius, and velocity
        :param other:
        :return:
        """
        return all(self.pos == other.pos) and all(self.v == other.v) \
            and self.r == other.r
