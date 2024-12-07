"""
Some sort of implementation of a spatial hash grid used to separate the
balls in the simulation into smaller grids, so that collisions do not need to be
checked for every ball. See e.g.:
https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/
spatial-hashing-r2697/
"""

import math
import misc
import numpy as np

from ball import Ball


class HashGrid:
    def __init__(self, w: int | float, h: int | float, cell_w: int | float,
                 cell_h: int | float, balls: list[Ball]) -> None:
        """
        :param w: Width of the total region containing the grid [m]
        :param h: Height of the total region containing the grid [m]
        :param cell_w: Width of a single cell [m]
        :param cell_h: Height of a single cell [m]
        :param balls: List of the balls that are to be placed into the grid
        """
        self.w = w
        self.h = h
        self.cw = cell_w
        self.ch = cell_h
        self.balls = balls
        self.grid = self._init_grid()

    def _init_grid(self) -> dict:
        """
        Initialises the grid with empty cells
        :return:
        """
        grid = {}
        # Determine the number of cells in each direction
        n_cells_x = math.ceil(self.w / self.cw)
        n_cells_y = math.ceil(self.h / self.ch)
        # Construct the grid row by row
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                grid[j, i] = []

        return grid

    def _hash(self, pos: np.ndarray) -> tuple[int, int]:
        """
        Computes the id for the cell that the ball is located
        :param pos: Position of the ball [m]
        :return:
        """
        # The ball might go out of bounds due to the discrete timestep
        # so let"s do a bit of a check to negate that
        min_x = min_y = 0.001
        max_x, max_y = self.w - 0.001, self.h - 0.001
        x = int(misc.clamp(a=min_x, b=max_x, num=pos[0]) / self.cw)
        y = int(misc.clamp(a=min_y, b=max_y, num=pos[1]) / self.ch)
        return y, x

    def add_ball(self, ball: Ball) -> None:
        """
        Adds the balsl to the grid based on their location. Key thing to note is that
        the ball is added only in to the one cell where its center is located.
        :param ball:
        :return:
        """
        self.grid[self._hash(ball.pos)].append(ball)

    def update_grid(self) -> None:
        """
        As the balls" position changes, they might move into a different grid cell.
        This function updates the balls" grid cells based on their new positions.
        :return:
        """
        # Clear the grid cells
        for j, i in self.grid.keys():
            self.grid[j, i] = []
        # Add the balls to their new locations
        for b in self.balls:
            self.add_ball(b)

    def get_nearby_balls(self, ball: Ball) -> list[Ball]:
        """
        Returns the balls that are in the same or in the neighboring cells
        with the given ball
        :param ball:
        :return:
        """
        balls = []
        r, c = self._hash(ball.pos)
        for j in range(r - 1, r + 2):
            for i in range(c - 1, c + 2):
                # If the search goes out of bounds, we"ll get a KeyError
                try:
                    balls.extend(self.grid[j, i])
                except KeyError:
                    pass

        # Let"s remove the ball passed as a parameter from the list so it won"t
        # collide with itself
        balls.remove(ball)
        return balls
