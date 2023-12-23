"""
Simulation of billiard-like balls moving around in a 2d box subject
to gravity. The animation is done with matplotlib.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ball import Ball
from time import perf_counter
from typing import Callable, Any
from common.hashgrid import HashGrid
from matplotlib.animation import FuncAnimation


def timer(f: Callable) -> Any:
    """
    Decorator to time a function
    :param f:
    :return:
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        s = perf_counter()
        rv = f(*args, **kwargs)
        print(f"Function '{f.__name__}' ran in {perf_counter() - s:.3f} s")
        return rv
    return wrapper


def _rand_range(a: int | float, b: int | float) -> int | float:
    """
    Return a random float in the range [a, b)
    :param a:
    :param b:
    :return:
    """
    return np.random.random() * (b - a) + a


def init_balls(n: int, v_max: int | float, r: int | float, w: int | float,
               h: int | float, c: tuple= (255, 0, 255)) -> list:
    """
    Creates a list of balls at random locations within the 2d box
    :param n: Number of balls
    :param v_max: Maximum velocity for a ball [m/s]
    :param r: Radius of the balls [m/s]
    :param w: Width of the box [m]
    :param h: Height of the box [m]
    :param c: Color of the balls
    :return:
    """
    balls = []
    xmin, xmax, ymin, ymax = 0, w, 0, h
    for _ in range(n):
        x = _rand_range(xmin, xmax)
        y = _rand_range(ymin, ymax)
        angle = _rand_range(0, 2 * np.pi)
        v_mag = _rand_range(0, v_max)
        v0 = np.array([np.cos(angle), np.sin(angle)]) * v_mag
        pos = np.array([x, y])
        balls.append(Ball(p0=pos, v0=v0, r=r, color=c))
    return balls


@timer
def simulate(grid: HashGrid, balls: list[Ball], dt: int | float, end: int | float,
             w: int | float, h: int | float) -> None:
    """
    Simulates the motion of the balls
    :param grid: A HashGrid containing da balls
    :param balls: List of the balls
    :param dt: Timestep [s]
    :param end: End time [s]
    :param w: Width of the box [m]
    :param h: Height of the box [m]
    :return:
    """
    for i in range(int(end / dt)):
        grid.update_grid()
        # Create the Path object for every ball
        for b in balls:
            others = grid.get_nearby_balls(ball=b)
            b.step_forward(others=others, dt=dt, w=w, h=h)


def _update_plots(n: int, axis: plt.axes, balls: list[Ball],
                  w: int | float, h: int | float, r: int | float) -> tuple:
    """
    Sets the data for the line-objects to be the coordinates
    at the current timestep for the animation
    :param n: Timestep
    :param axis: A plt.axes object for which to draw stuff
    :param balls: List of the balls
    :param w: Width of the box [m]
    :param h: Height of the box [m]
    :param r: Radius of the ball [m]
    :return:
    """
    axis.clear()
    axis.set_xlim(0, w)
    axis.set_ylim(0, h)
    circles = [plt.Circle(xy=(b.positions[n, 0], b.positions[n, 1]), radius=r,
                          linewidth=0) for b in balls]
    cs = [b.color for b in balls]
    c_patches = mpl.collections.PatchCollection(circles, facecolor=cs)
    c_patches.set_paths(c_patches)
    axis.add_collection(c_patches)
    return axis,


def _init_anim(fig: plt.figure, ax: plt.axes, balls: list[Ball], w: int,
               h: int, r: int | float) -> tuple:
    """
    :param fig:
    :param ax:
    :param balls:
    :param w:
    :param h:
    :param r:
    :return:
    """
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    size = (w + h) / 2
    ms = 2 * r * ax.get_window_extent().width / size * 72 / fig.dpi
    xs = [b.positions[0, 0] for b in balls]
    ys = [b.positions[0, 1] for b in balls]
    # vels = [np.linalg.norm(b.vels[0]) for b in balls]
    colors = [b.positions[0, 1] for b in balls]
    plots = ax.scatter(xs, ys, marker="o", s=ms, c=colors, cmap="jet")
    return plots


def _animate(n: int, ax: plt.axes, balls: list[Ball]) -> tuple:
    """
    :param n:
    :param ax:
    :param balls:
    :return:
    """
    coords = [b.positions[n] for b in balls]
    # colors = [np.linalg.norm(b.vels[n]) for b in balls]
    ax.set_offsets(coords)
    # ax.set_array(colors)
    return ax,


@timer
def create_anim(balls: list[Ball], dt: int | float, end: int | float,
                w: int | float, h: int | float, r: int | float) -> None:
    """
    Creates an animation using the data from the simulated motion of the balls
    :param balls: List of the balls
    :param dt: Timestep [s]
    :param end: Ending time [s]
    :param w: Width of the box [m]
    :param h: Height of the box [m]
    :param r: Radius of the balls [m]
    :return:
    """
    timesteps = int(end / dt)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = _init_anim(fig=fig, ax=ax, balls=balls, w=w, h=h, r=r)
    anim = FuncAnimation(fig=fig, func=_animate,
                         frames=timesteps, fargs=(ax, balls), blit=True)
    anim.save(filename="simulation.gif", writer="pillow", fps=30, dpi=100)


def main():
    # Initial conditions
    width, height = 2, 2  # Dimensions of the 2d box [m]
    # TODO: Investigate why "x not in list" happens (large velocity and timestep)
    dt = 0.05  # Timestep [s]
    end = 1  # End time of the simulation [s]
    num_balls = 1000  # Number of balls
    radius = 0.025  # [m]
    v_max = 5  # Maximum velocity for a ball [m/s]
    cell_w = cell_h = 4 * radius  # Size of a cell in the hash grid [m]

    # Simulation and animation
    balls = init_balls(n=num_balls, v_max=v_max, r=radius, w=width, h=height)
    grid = HashGrid(balls=balls, w=width, h=height, cell_w=cell_w, cell_h=cell_h)
    simulate(grid=grid, balls=balls, dt=dt, end=end, w=width, h=height)
    create_anim(balls=balls, dt=dt, end=end, w=width, h=height, r=radius)


if __name__ == "__main__":
    main()
