"""
This module implements a generic Runge-Kutta solver.
"""
import numpy as np


def runge_kutta(f, x0, t_start, t_end, delta_t):
    """
    The Runge-Kutta method for `n` dimensional (non-linear) systems.

    Parameters
    ----------
    f: function
        A list containing the entries of the vector-valued function `f`.
    x0: list
        Contains the initial values of the system.
    t_start: float
        The starting time.
    t_end: float
        The ending time.
    delta_t: float
        The time step used when solving the system.

    Return
    ------
    tuple
        The first entry is a list containing all the employed time steps.
        The second entry is a list, containing the corresponding at each time step.
    """
    t = []
    x = []
    t_last = t_start
    x_last = x0
    while t_last < t_end:
    
        k1 = f(x_last) * delta_t
        k2 = f(x_last + k1 * 0.5) * delta_t
        k3 = f(x_last + k2 * 0.5) * delta_t
        k4 = f(x_last + k3) * delta_t

        x_last = x_last + 1. / 6. * (k1 + k2 * 2. + k3 * 2 + k4)
        x.append(x_last)
        t_last = t_last + delta_t
        t.append(delta_t)

    return np.array(t), np.array(x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
