"""
File for the Ball-class used in the simulation
"""

from __future__ import annotations

import misc
import numpy as np


class Ball:
    def __init__(self, p0: np.ndarray, v0: np.ndarray, r: int | float,
                 color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """
        :param p0: Initial position of the ball [m]
        :param v0: Initial velocity of the ball [m/s]
        :param r: Radius of the ball [m]
        :param color: Optional parameter for the color of the ball that will be
            used in the animation. Must be given as an RGB tuple of ints in the
            range 0 ... 255. Defaults to black (0, 0, 0).
        """
        self.pos = p0
        self.v = v0
        self.r = r
        # The colors must be in range 0 ... 1
        self.color = [c / 255 for c in color]
        # Create the arrays for the positions and velocities throughout the
        # simulation
        self.positions = np.zeros(shape=(1, 2))
        self.positions[0] = p0
        self.vels = np.zeros(shape=(1, 2))
        self.vels[0] = p0
        # In the beginning, the Ball does not have a Path yet
        self.path = None
        # Whether the ball has collided or not with another ball during
        # the ongoing timestep
        self.has_collided = False

    def _apply_boundaries(self, w: int | float, h: int | float) -> None:
        """
        Handles the collision between the ball and a wall
        :param w: Width of the box [m]
        :param h: Height of the box [m]
        :return:
        """
        x, y = self.pos
        # Left boundary
        if (x - self.r) < 0:
            # Set the ball"s position within the boundaries
            self.pos[0] = self.r
            # Flip the horizontal velocity
            self.v[0] = -self.v[0]
        # Right boundary
        if (x + self.r) > w:
            # Set the ball"s position within the boundaries
            self.pos[0] = w - self.r
            # Flip the horizontal velocity
            self.v[0] = -self.v[0]
        # Bottom boundary
        if (y - self.r) < 0:
            # Set the ball"s position within the boundaries
            self.pos[1] = self.r
            # Flip the vertical velocity
            self.v[1] = -self.v[1]
        # Top boundary
        if (y + self.r) > h:
            # Set the ball"s position within the boundaries
            self.pos[1] = h - self.r
            # Flip the vertical velocity
            self.v[1] = -self.v[1]

    def _handle_collision(self, other: Ball, direc: np.ndarray,
                          dist: int | float) -> None:
        """
        :param other:
        :param direc:
        :param dist:
        :return:
        """
        # Move the bodies so that they don't overlap anymore
        direc *= (1 / dist)
        corr = (self.r + other.r - dist) / 2
        self.pos += direc * -corr
        other.pos += direc * corr
        # Calculate the end velocities for the both balls
        v1, v2 = self.v, other.v
        v12 = v1 - v2
        v21 = v2 - v1
        r12 = self.pos - other.pos
        r21 = other.pos - self.pos
        self.v = v1 - (np.dot(v12, r12) / misc.sq_len(r12)) * r12
        other.v = v2 - (np.dot(v21, r21) / misc.sq_len(r21)) * r21
        # Create the Paths for the two balls
        # Mark that the balls collided on this timestep
        self.has_collided = True
        other.has_collided = True

    def _apply_collisions(self, others: list[Ball]) -> None:
        """
        Applies collisions between the balls. As of right now, only takes into
        account a single collision, i.e., a collision with only one other ball.
        :param others: List of all the balls the ball under inspection could
            collide
        :return:
        """
        for other in others:
            # If the ball has already collided, do nothing
            if other.has_collided:
                continue
            direc = other.pos - self.pos
            dist = misc.vec_len(direc)
            rsum = self.r + other.r
            if dist <= rsum:
                self._handle_collision(other=other, direc=direc, dist=dist)
                # We assume that the ball collides with only one other ball
                return

    def step_forward(self, others: list[Ball], dt: int | float, w: int | float,
                     h: int | float) -> None:
        """
        :param others: The "other" balls that our ball can collide with
        :param dt: Timestep [s]
        :param w: Width of the box [m]
        :param h: Height of the box [m]
        :return:
        """
        # Compute new velocities (and positions) due to collisions
        self._apply_collisions(others=others)
        self._apply_boundaries(w=w, h=h)
        # Take a step forward in time
        self.pos += self.v * dt
        self.positions = np.vstack(tup=(self.positions, self.pos))
        self.vels = np.vstack(tup=(self.vels, self.v))
        # Reset the collision for the next timestep
        self.has_collided = False

    def __eq__(self, other: Ball) -> bool:
        """
        Two balls are equal if they have the same position, radius, and velocity
        :param other:
        :return:
        """
        return all(self.pos == other.pos) and all(self.v == other.v) \
            and self.r == other.r
