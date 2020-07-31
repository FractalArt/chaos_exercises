"""
This script generates the plots to provide graphs helping to solve exercise 3.4.2.

The differential equation is given by:

x' = r*x - sinh(x)

An obvious fixed point is x* = 0 and the remaining piece of the bifurcation diagram
can be drawn by inverting the plot of r(x) = sinh(x) / x, albeit without the stability
information.
"""
import matplotlib.pyplot as plt
import numpy as np


def r(x):
    """
    Express r in terms of the fixed point x*.

    This is done to obtain the bifurcation diagram for x* != 0.
    """
    return np.sinh(x) / x


if __name__ == "__main__":
    # Plot r as a function of the fixed points
    # Inverting the axes of this plot, we obtain the bifurcation
    # diagram, albeit without the stability information (although,
    # if we know the general appearance of sub and supercritcal
    # pitchfork bifurcations, we can infer the stability of the
    # fixed points from the information of whether they appear
    # to the right of the bifurcation (stable) or to the left (unstable)).
    x_vals = np.linspace(-1, 1, 1000)
    r_vals = r(x_vals)

    plt.plot(x_vals, r_vals)
    plt.title(r"$r$ as a function of the fixed points $x^*$")
    plt.xlabel(r"$x^*$")
    plt.ylabel(r"$r(x^*)$")
    plt.show()

    # Compare rx and sinh(x) to determine the fixed points (from their intersection)
    # as well as their stability.

    x_vals = np.linspace(-3, 3, 1000)

    def rx(x, r):
        return x*r

    plt.plot(x_vals, np.sinh(x_vals), label=r"$\sinh(x)$")
    plt.plot(x_vals, rx(x_vals, -1), label=r"r=-1")
    plt.plot(x_vals, rx(x_vals, 0), label=r"r=0")
    plt.plot(x_vals, rx(x_vals, 1), label=r"r=1")
    plt.plot(x_vals, rx(x_vals, 2), label=r"r=2")
    plt.title(r"$rx$ vs $\sinh(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()
