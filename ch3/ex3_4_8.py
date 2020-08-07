#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def f(x, r):
    """
    Implement the right-hand-side of the differential equation

        x' = r * x - x / (1 + x)
    """
    return r * x - x / (1 + x**2)


def plot(x, y, r):
    """
    Plot the vector field for a given value of the parameter `r`.

    Parameters
    ----------
    x: np.array
        The x-values
    y: np.array
        The y-values obtained for a specifc value of `r`
    r: float
        The value of `r` for which the `y` values have been obtained from `x`.
    """
    plt.plot(x, np.zeros(len(y)), color='grey')
    plt.plot(x, y)
    plt.xlabel('$x$')
    plt.ylabel(r'$\dot{x}$')
    plt.title(r'$\dot{x} = $' + f'$f(x, r={r})$')
    plt.show()


if __name__ == "__main__":
    x = np.linspace(-3., 3., 99)

    # Create a plot of the vector field for different values of r
    for r in [-2, 0, 0.5, 1, 5]:
        y = f(x, r)
        plot(x, y, r)


    
