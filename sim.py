"""
Simulation of thermal motion of particles contained in
a 2D box or some shiieeeeett
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from copy import copy
from particle import Particle
from matplotlib.animation import FuncAnimation

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\henri\ffmpeg\bin' \
                                        r'\ffmpeg.exe'

# Constants
k = 1.390658e-23  # Boltzmann's constant [J/K]
u = 1.6605402e-27  # Atomic mass unit [kg]

# Shortcuts for type hinting
numeric = int | float


def _rand_range(a: numeric, b: numeric) -> numeric:
    """
    Return a random float in the range [a, b)
    :param a:
    :param b:
    :return:
    """
    return np.random.random() * (b - a) + a


def init_particles(n: int, mass: numeric, v_mag: numeric, w: numeric,
                   h: numeric) -> list:
    """
    Creates a list of particles at random locations within the 2D box
    :param n: Number of particles
    :param mass: Mass of the particles
    :param v_mag: Initial magnitude for the particle's velocities
    :param w: Width of the box
    :param h: Height of the box
    :return:
    """
    parts = []
    xmin, xmax, ymin, ymax = 0, w, 0, h
    for _ in range(n):
        x = _rand_range(xmin, xmax)
        y = _rand_range(ymin, ymax)
        angle = _rand_range(0, 2 * np.pi)
        v0 = np.array([np.cos(angle), np.sin(angle)]) * v_mag
        pos = np.array([x, y])
        parts.append(Particle(pos0=pos, v0=v0, mass=mass))
    return parts


def _plot_line(p0: tuple, p1: tuple, **kwargs) -> plt.Line2D:
    """
    Plots a line between two points
    :param p0: Starting point
    :param p1: Ending point
    :param kwargs: Keyword arguments accepted by the plt.plot-function
    :return:
    """
    line, = plt.plot([p0[0], p1[0]], [p0[1], p1[1]], **kwargs)
    return line


def _plot_2dbox(w: numeric, h: numeric, **kwargs) -> list:
    """
    Plots a 2D box with the bottom right corner at coordinates (0, 0)
    :param w: Width of the box
    :param h: Height of the box
    :param kwargs: Keyword arguments accepted by matplotlib's
    plot-function
    :return: List of matplotlib line-objects
    """
    line1 = _plot_line((0, 0), (w, 0), **kwargs)
    line2 = _plot_line((w, 0), (w, h), **kwargs)
    line3 = _plot_line((w, h), (0, h), **kwargs)
    line4 = _plot_line((0, h), (0, 0), **kwargs)
    return [line1, line2, line3, line4]


def simulate(particles: list, dt: numeric, end: numeric, epsilon: numeric,
             w: numeric, h: numeric) -> np.ndarray:
    """
    Simulates the motion of the particles
    :param particles: A list of Particle-updates
    :param dt: Timestep
    :param end: End time
    :param epsilon: If the distance between particles is below this value,
        the particles are deemed to collide. Essentially the sum of the particles'
        radiuses.
    :param w: Width of the box
    :param h: Height of the box
    :return: A list of the locations of each particle at each timestep
    """
    timesteps = int(end / dt)
    coords = np.zeros((len(particles), timesteps, 2))
    for i in range(timesteps):
        for j, p in enumerate(particles):
            others = copy(particles)
            others.remove(p)
            p.update_position(others=others, dt=dt, epsilon=epsilon, w=w, h=h)
            coords[j, i] = p.pos
    return coords


def _create_plots(data: np.ndarray, w: numeric, h: numeric,
                  box_kwargs: dict = None, path_kwargs: dict = None) -> tuple:
    """
    Creates the plot-objects for the particles
    :param data: List of the arrays containing the coordinates
    for the particles
    :param w: Width of the box
    :param h: Height of the box
    :param box_kwargs: Kwargs for the box, must be accepted by
    the plt.plot-function
    :param path_kwargs: Kwargs for the particles, must be accepted by
    the plt.plot-function
    :return:
    """
    box = _plot_2dbox(w, h, **box_kwargs)
    paths = [plt.plot(coords[:, 0], coords[:, 1], **path_kwargs)[0]
             for coords in data]
    return box, paths


def _update_plots(n: int, paths: list, data: list) -> list:
    """
    Sets the data for the line-objects to be the coordinates
    at the current timestep for the animation
    :param n: Timestep
    :param paths: List of the line-objects
    :param data: List of the coordinates of the particles
    :return:
    """
    for path, coords in zip(paths, data):
        path.set_data(coords[n, :])
    return paths


def animate(data: np.ndarray, dt: numeric, end: numeric, w: numeric, h: numeric,
            box_kwargs: dict = None, path_kwargs: dict = None) -> None:
    """
    Plot the locations of the particles as an animation
    :param data: Location of each particle at each timestep
    :param dt: Timestep
    :param end: Ending time
    :param w: Width of the box
    :param h: height of the box
    :param box_kwargs: Kwargs for the box, must be accepted by
    the plt.plot-function
    :param path_kwargs: Kwargs for the particles, must be accepted by
    the plt.plot-function
    :return:
    """
    timesteps = int(end / dt)
    fig = plt.figure()
    plt.autoscale()
    plt.grid()
    _, paths = _create_plots(data, w, h, box_kwargs, path_kwargs)
    anim = FuncAnimation(fig, _update_plots, timesteps,
                         fargs=(paths, data), interval=10, blit=False,
                         repeat=False)
    writer = animation.FFMpegWriter(fps=30)
    anim.save('brownian.avi', writer=writer)
    plt.show()


def main():
    width, height = 5, 5  # [m]
    dt, end = 0.0001, 0.1  # [s]
    particle_mass = 32 * u  # [kg]
    num_particles = 50
    epsilon = 1e-1  # [m]
    v0_mag = 300  # [m/s]
    particles = init_particles(n=num_particles, mass=particle_mass, v_mag=v0_mag,
                               w=width, h=height)
    data = simulate(particles, dt, end, epsilon=epsilon, w=width, h=height)
    box_kws = {'color': 'blue'}
    path_kws = {'color': 'red', 'marker': 'o', 'lw': 1}
    animate(data, dt, end, width, height, box_kws, path_kws)


if __name__ == '__main__':
    main()
