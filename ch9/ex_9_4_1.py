"""
Exercise 9.4.1: Computer work.

Using numerical integration, compute the Lorenz map for
r=28, sigma=10, b=8/3.
"""
import argparse as ap
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import numpy as np

from typing import Callable


def cli():
    """Setup the command-line interface."""
    parser = ap.ArgumentParser()
    parser.add_argument('-n', '--n-steps', type=int, default=100000,
                        help='The of time steps to perform the integration.')
    parser.add_argument('-t', '--time-step', type=float, default=0.001,
                        help='The time step to be used in the numerical integration.')
    return parser.parse_args()


def lorenz(x, s=10, b=8./3., r=28.):
    """The Lorenz system with standard parameters for sigma, b and r."""
    return np.array([
        s * (x[1] - x[0]),
        r * x[0] - x[1] - x[0] * x[2],
        x[0] * x[1] - b * x[2]
    ])


def runge_kutta(f: Callable, delta_t: float, initial: np.array):
    k1 = f(initial) * delta_t
    k2 = f(initial + 0.5 * k1) * delta_t
    k3 = f(initial + 0.5 * k2) * delta_t
    k4 = f(initial + k3) * delta_t
    return initial + 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


if __name__ == "__main__":

    # parse the command line arguments
    args = cli()

    # define the initial condition
    initial_condition = np.array([1.0, 1.0, 1.0])

    # compute the Lorenz map by storing the values with time
    z_values = []
    times = []

    previous = initial_condition
    current_time = 0

    for _ in range(args.n_steps):
        z_values.append(previous[-1])
        times.append(current_time)
        current_time += args.time_step
        previous = runge_kutta(lorenz, args.time_step, previous)

    # compute the indices of the maxima in the sequence
    z_max_idx = argrelmax(np.array(z_values))[0]

    z_max = []
    t_max = []

    for idx in z_max_idx:
        z_max.append(z_values[idx])
        t_max.append(times[idx])

    # check whether the maxima are determined correctly by plotting them
    # on top of the time evolution of z
    plt.plot(times, z_values, color='blue')
    plt.scatter(t_max, z_max, color='red')
    plt.xlabel('t')
    plt.ylabel('z(t)')
    plt.show()

    plt.plot(z_max[:-1], z_max[:-1], color='black', linestyle=':')
    plt.scatter(z_max[:-1], z_max[1:], color='blue', s=1)
    plt.title('Lorenz Map')
    plt.xlabel(r'$z_n$')
    plt.ylabel(r'$z_{n+1}$')
    plt.show()
