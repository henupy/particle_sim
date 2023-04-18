"""
Simulation of billiard-like balls moving around in a 2d box (not subject
to gravity). The animation is done with matplotlib.
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
               h: int | float) -> list:
    """
    Creates a list of balls at random locations within the 2d box
    :param n: Number of balls
    :param v_max: Maximum velocity for a ball [m/s]
    :param r: Radius of the balls [m/s]
    :param w: Width of the box [m]
    :param h: Height of the box [m]
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
        balls.append(Ball(p0=pos, v0=v0, r=r))
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
            b.create_path(others=others, dt=dt, w=w, h=h)
        # Execute the Path of each ball, i.e., update their positions
        for b in balls:
            b.step_forward()


def _update_plots(n: int, axes: list[plt.axes], balls: list[Ball],
                  w: int | float, h: int | float, r: int | float) -> None:
    """
    Sets the data for the line-objects to be the coordinates
    at the current timestep for the animation
    :param n: Timestep
    :param axes: A list of two plt.axes objects. The first one must be for the
    scatter plot of the balls and the second one for the plot of the velocity
    distribution.
    :param balls: List of the balls
    :param w: Width of the box [m]
    :param h: Height of the box [m]
    :param r: Radius of the ball [m]
    :return:
    """
    # Clear both axes to set up a new frame
    [ax.clear() for ax in axes]

    # Plot the balls on the first axis object
    ax = axes[0]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    circles = [plt.Circle(xy=(b.positions[n, 0], b.positions[n, 1]), radius=r,
                          linewidth=0) for b in balls]
    cs = [b.color for b in balls]
    c_patches = mpl.collections.PatchCollection(circles, facecolor=cs)
    ax.add_collection(c_patches)

    # Plot the velocity distribution to the other axis
    ax = axes[1]
    vels = [b.vels[n] for b in balls]
    vel_mag = np.sqrt(np.sum(np.power(vels, 2), axis=1))
    vel_max = np.max(vel_mag)
    vel_avg = np.mean(vel_mag)
    bins = np.linspace(0, vel_max * 1.1, 50)
    ax.hist(vel_mag, bins=bins, density=True)
    v = np.linspace(0, vel_max * 1.1, 1000)
    a = 2 / (vel_avg * vel_avg)
    fv = a * v * np.exp(-a * np.power(v, 2) / 2)
    ax.plot(v, fv)
    ax.set_xlabel('Velocity [m/s]')
    ax.set_ylabel('Number of particles')
    ax.set_xlim(0, vel_max * 1.1)
    ax.set_ylim(0, 0.025)


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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    anim = FuncAnimation(fig=fig, func=_update_plots, frames=timesteps,
                         fargs=(axes, balls, w, h, r), interval=1)
    anim.save(filename='simulation.gif', writer='pillow', fps=30, dpi=100)


def main():
    # Initial conditions
    width, height = 1, 1  # Dimensions of the 2d box [m]
    dt = 0.00005  # Timestep [s]
    end = 0.005  # End time of the simulation [s]
    num_balls = 200  # Number of balls
    radius = 0.01  # [m]
    v_max = 200  # Maximum velocity for a ball [m/s]
    cell_w = cell_h = 4 * radius  # Width and height of a cell in the hash grid [m]

    # Simulation and animation
    balls = init_balls(n=num_balls, v_max=v_max, r=radius, w=width, h=height)
    grid = HashGrid(balls=balls, w=width, h=height, cell_w=cell_w, cell_h=cell_h)
    simulate(grid=grid, balls=balls, dt=dt, end=end, w=width, h=height)
    create_anim(balls=balls, dt=dt, end=end, w=width, h=height, r=radius)


if __name__ == '__main__':
    main()
