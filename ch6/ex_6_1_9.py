"""
Exercise 6.1.9: Dipole fixed point
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from phase_plane.runge_kutta import runge_kutta


def f(v):
    """
    The right-hand side of the system of differential equations.

    ..math:: \dot{x} = 2xy, \dot{y} = y^2-x^2
    """
    x = v[0]
    y = v[1]
    return np.array([2 * x * y, y**2 - x**2])


def setup_cli(args):
    """
    Setup up command-line parser.
    """
    parser = argparse.ArgumentParser(description='Exercise 6.1.8: Van der Pol oscillator.')
    parser.add_argument('-p', type=int, default=15, help='The number of points to use in the x and y direction.')
    parser.add_argument('--x_min', type=float, default=-3.5, help='Minimal value of x to be plotted.')
    parser.add_argument('--x_max', type=float, default=+3.5, help='Minimal value of x to be plotted.')
    parser.add_argument('--y_min', type=float, default=-3.5, help='Minimal value of y to be plotted.')
    parser.add_argument('--y_max', type=float, default=+3.5, help='Minimal value of y to be plotted.')
    parser.add_argument('--delta_t', type=float, default=0.01, help='Time step used in the numerical solution of the DE.')
    parser.add_argument('--end_t', type=float, default=1.0, help='Final time in the solution of the DE (Starting from t=0).')
    return parser.parse_args(args)

if __name__ == "__main__":
    # Parse command-line arguments
    args = setup_cli(sys.argv[1:])

    # Draw vector field
    x, y = np.meshgrid(np.linspace(args.x_min, args.x_max, args.p), np.linspace(args.y_min, args.y_max, args.p))
    u = 2*x*y
    v = y**2 - x**2

    plt.title('Dipole Fixed Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(args.x_min, args.x_max)
    plt.ylim(args.y_min, args.y_max)
    plt.quiver(x, y, u, v)

    # Draw phase portraits
    x0 = [0.5, 0.5]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C1", label=f'x0={x0}')

    x0 = [-3., -3.0]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C2", label=f'x0={x0}')

    x0 = [-2.5, -1.5]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C3", label=f'x0={x0}')

    x0 = [-2.5, -1.]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C4", label=f'x0={x0}')

    x0 = [-2.5, 1.]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C5", label=f'x0={x0}')

    x0 = [0.5, 0.5]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C6", label=f'x0={x0}')

    # Plot everything
    plt.legend()
    plt.show()

