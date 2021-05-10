import argparse as ap
import matplotlib.pyplot as plt
import numpy as np


def cli():
    """Setup the command-line interface"""
    parser = ap.ArgumentParser()
    parser.add_argument('-x', '--x', type=lambda s: float(s), help='The initial x value')
    parser.add_argument('-y', '--y', type=lambda s: float(s), help='The initial y value')
    parser.add_argument('-z', '--z', type=lambda s: float(s), help='The initial z value')
    parser.add_argument('-s', '--sigma', type=float, default=10., help='The Prantl number in the Lorenz equations')
    parser.add_argument('-b', '--b', type=float, default=8./3., help='The parameter in the Lorenz equations')
    parser.add_argument('-r', '--r', type=float, help='The r parameter in the Lorenz equations')
    parser.add_argument('-n', '--n-steps', type=int, default=10000, help='The number of steps to perform in the numerical integration')
    parser.add_argument('-t', '--time-step', type=float, default=0.01, help='The time step to use in the numerical integration')
    parser.add_argument('--skip', type=int, default=0, help='The number of transient points in the beginning to skip')
    return parser.parse_args()


def lorenz(x, s, b, r):
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

    args = cli()

    assert(args.sigma > 0)
    assert(args.r > 0)
    assert(args.b > 0)
    assert(args.time_step > 0)
    assert(args.n_steps > 0)

    print(f"sigma: {args.sigma}")
    print(f"b    : {args.b}")
    print(f"r    : {args.r}")

    initial = np.array([args.x, args.y, args.z])
    print(f"initial : {initial}")

    previous = initial
    points = [initial]
    times = [0]
    running_time = 0

    for _ in range(args.n_steps):
        running_time += args.time_step
        previous = runge_kutta(lambda x: lorenz(x, args.sigma, args.b, args.r),
                               args.time_step, previous)
        points.append(previous)
        times.append(running_time)

    points = np.stack(points, axis=1)
    print(points.shape)
    print(points[1, :])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(times[args.skip:], points[0, args.skip:], color='C1')
    ax1.set_xlabel('t [AU]')
    ax1.set_ylabel('x')

    ax2.plot(times[args.skip:], points[1, args.skip:], color='C2')
    ax2.set_xlabel('t [AU]')
    ax2.set_ylabel('y')

    ax3.scatter(points[0, args.skip:], points[2, args.skip:], color='C3', s=0.3, marker='.')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')

    plt.show()

