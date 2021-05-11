"""
Exercise 9.3.9: Exponential divergence.

Determine the largest Liapunov exponent of the Lorenz equations
for the standard parameter values:
    r = 28, sigma = 10, b = 8/3
"""
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np


def cli():
    """Setup the command-line interface."""
    parser = ap.ArgumentParser()
    parser.add_argument('-n', '--n-steps', type=int, default=100000,
                        help='The of time steps to perform the integration.')
    parser.add_argument('-t', '--time-step', type=float, default=0.001,
                        help='The time step to be used in the numerical integration.')
    parser.add_argument('--transient-steps', type=int, default=1000,
                        help='The number of steps to perform to let the state reach the attractor.')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot delta.')
    return parser.parse_args()


def lorenz(x, s=10, b=8./3., r=28.):
    """The Lorenz system with standard parameters for sigma, b and r."""
    return np.array([
        s * (x[1] - x[0]),
        r * x[0] - x[1] - x[0] * x[2],
        x[0] * x[1] - b * x[2]
    ])


def runge_kutta(f, delta_t, initial):
    k1 = f(initial) * delta_t
    k2 = f(initial + 0.5 * k1) * delta_t
    k3 = f(initial + 0.5 * k2) * delta_t
    k4 = f(initial + k3) * delta_t
    return initial + 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


if __name__ == "__main__":

    # parse the command line arguments
    args = cli()

    # define two initial conditions that are close in phase space
    # only different by a small amount in the z component
    x = 10.
    y = 10.
    z = 10.
    delta = 1e-14

    p_1 = np.array([x, y, z])

    current_time = 0
    deltas = []
    times = []

    # first, let the point reach the attractor
    for _ in range(args.transient_steps):
        p_1 = runge_kutta(lambda x: lorenz(x), args.time_step, p_1)

    # now that the attractor is reached, apply a perturbation
    p_2 = np.array([p_1[0], p_1[1], p_1[2] + delta])

    # the initial difference between the two points on the attractor
    previous_delta = np.linalg.norm(p_1 - p_2)

    # let the two points evolve on the attractor
    for _ in range(args.n_steps):
        current_time += args.time_step
        times.append(current_time)

        p_1 = runge_kutta(lambda x: lorenz(x), args.time_step, p_1)
        p_2 = runge_kutta(lambda x: lorenz(x), args.time_step, p_2)

        previous_delta = np.linalg.norm(p_1 - p_2)
        deltas.append(previous_delta)

    # compute the logarithm of the delta values
    log_deltas = np.log(np.array(deltas))
    times = np.array(times)

    # plot the logarithm of delta to be able to determine in which region to perform
    if not args.no_plot:
        plt.plot(times, log_deltas)
        plt.show()

    # perform the plot in the range 0 to 15'000
    n = 20000

    intercept = log_deltas[0]
    slope = (log_deltas[n] - log_deltas[0]) / (times[n] - times[0])

    print(f"intercept: {intercept}")
    print(f"slope    : {slope}")

    if not args.no_plot:
        plt.plot(times, log_deltas)
        fitted = slope * times + intercept
        plt.plot(times, fitted)
        plt.show()
