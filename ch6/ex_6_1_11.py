"""
Exercise 6.1.8: Parrot
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from phase_plane.runge_kutta import runge_kutta


def f(v):
    """
    The right-hand side of the Parrot system.

    ..math:: \dot{x} = y, \dot{y} = -x + y (1 - x^2)
    """
    x = v[0]
    y = v[1]
    return np.array([y + y**2, -x + y / 5. - x * y + 6. / 5. * y**2])


if __name__ == "__main__":
    x_min = -10
    x_max = +10
    y_min = -10
    y_max = +10

    # Draw vector field
    x, y = np.meshgrid(np.linspace(x_min, x_max, 35), np.linspace(y_min, y_max, 35))
    u = y + y**2
    v = -x + y / 5. - x * y + 6. / 5. * y**2

    plt.title('Parrot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v)

    # Draw phase portraits
    x0 = [-3., -3.0]
    t, xy = runge_kutta(f, x0, 0, 100, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C2", label=f'x0={x0}')

    x0 = [0.5, 0.5]
    t, xy = runge_kutta(f, x0, 0, 10, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C1", label=f'x0={x0}')

    x0 = [-8.5, -2.5]
    t, xy = runge_kutta(f, x0, 0, 10, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C5", label=f'x0={x0}')

    x0 = [-9.5, -9.5]
    t, xy = runge_kutta(f, x0, 0, 10, 0.01)
    x, y = xy.T[0], xy.T[1]
    plt.plot(x, y, c=f"C6", label=f'x0={x0}')

    # Plot nullcline
    z = np.linspace(x_min, x_max, 100)
    plt.plot(z, np.zeros(len(z)), c=f"C3", label='Nullcline: $\dot{x} = 0$')

    # For the nullclines associated to y_dot, make sure we only consider
    # x-values in the domain of the function.
    exclude_low = (-11 - np.sqrt(120)) / 5
    exclude_high = (-11 + np.sqrt(120)) / 5

    x1 = np.linspace(x_min, exclude_low, 50)
    x2 = np.linspace(exclude_high, x_max, 50)

    nullcline_y_1 = 5. * ( (x1 - 1. / 5.) - np.sqrt((1. / 5. - x1)**2 + 4. * x1 * 6. / 5.) ) / 12.
    nullcline_y_2 = 5. * ( (x2 - 1. / 5.) - np.sqrt((1. / 5. - x2)**2 + 4. * x2 * 6. / 5.) ) / 12.
    nullcline_y_3 = 5. * ( (x1 - 1. / 5.) + np.sqrt((1. / 5. - x1)**2 + 4. * x1 * 6. / 5.) ) / 12.
    nullcline_y_4 = 5. * ( (x2 - 1. / 5.) + np.sqrt((1. / 5. - x2)**2 + 4. * x2 * 6. / 5.) ) / 12.
    plt.plot(x1, nullcline_y_1, c=f"C4", label='Nullcline: $\dot{y} = 0$')
    plt.plot(x2, nullcline_y_2, c=f"C4")
    plt.plot(x1, nullcline_y_3, c=f"C4")
    plt.plot(x2, nullcline_y_4, c=f"C4")

    # Plot everything
    plt.legend()
    plt.show()

